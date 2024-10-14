
import argparse
import os
import time

import cv2
import numpy as np
from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Translate into absolute screen coordinates.
        x, y = scale_x * x, scale_y * y
        w, h = scale_x * w, scale_y * h
        svg.add_bbox(x, y, w, h, obj.score)
        if obj.id in labels:
            svg.add_text(x, y - 10, labels[obj.id])

    return svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Path to the .tflite model file.')
    parser.add_argument('--labels', required=True,
                        help='Path to the labels file.')
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--video_file', help='Path to the input video file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save frames with detected objects', default='output_frames')
    parser.add_argument('--detect_object', help='Specify the object to detect (e.g., dog)', required=False)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)  # Expected input size, e.g., (300, 300)

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if specific object is set to be detected
    detect_object_id = None
    if args.detect_object:
        # Find the ID of the specified object
        detect_object_id = [id for id, label in labels.items() if label.lower() == args.detect_object.lower()]
        if not detect_object_id:
            print(f"Error: Object '{args.detect_object}' not found in the label file.")
            return
        detect_object_id = detect_object_id[0]
        print(f"Detecting object: {args.detect_object} (ID: {detect_object_id})")

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)

    # Open video file
    cap = cv2.VideoCapture(args.video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file.")
            break

        start_time = time.monotonic()

        # Get original frame size (e.g., 1920x1080)
        original_height, original_width, _ = frame.shape

        # Resize frame to the expected input size of the model (300x300)
        resized_frame = cv2.resize(frame, inference_size)

        # Convert the resized frame to a flat tensor of size 300x300x3
        input_tensor = np.asarray(resized_frame).flatten()

        run_inference(interpreter, input_tensor)

        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        end_time = time.monotonic()

        # Display results
        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))),
        ]
        print(' '.join(text_lines))

        # Save frames with detected objects
        save_frame = False
        for obj in objs:
            # If detect_object is specified, check if it matches the object
            if detect_object_id is not None and obj.id != detect_object_id:
                continue  # Skip if not the object we're looking for

            if obj.bbox.valid:
                save_frame = True
                # Calculate bounding box scaling to the original frame size (from 300x300 to 1920x1080)
                scale_x = original_width / inference_size[0]
                scale_y = original_height / inference_size[1]
                x_min = int(obj.bbox.xmin * scale_x)
                y_min = int(obj.bbox.ymin * scale_y)
                x_max = int(obj.bbox.xmax * scale_x)
                y_max = int(obj.bbox.ymax * scale_y)

                # Draw bounding box and label on the original frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                if obj.id in labels:
                    cv2.putText(frame, labels[obj.id], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame to the output directory if the object is detected
        if save_frame:
            output_path = os.path.join(args.output_dir, f'frame_{frame_count:05d}.jpg')
            cv2.imwrite(output_path, frame)
            print(f'Saved: {output_path}')

        frame_count += 1

    cap.release()
    print("Processing complete. Frames saved in:", args.output_dir)


if __name__ == '__main__':
    main()
