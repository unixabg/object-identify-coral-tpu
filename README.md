# object-identify-coral-tpu
Modified code to use with existing coral examples to accept video input and identify objects from videos

For testing I used a PI3 B+, a Coral USB Accelerator, and the recommended AIY Maker Kit system image from https://coral.ai/docs/accelerator/get-started/#requirements .

Below is some information and instructions I followed from https://github.com/google-coral/examples-camera/tree/master

This repo contains an example of detecting objects from mp4 video files
together with the [TensorFlow Lite API](https://tensorflow.org/lite) with a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board).

## Installation

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera.git --depth 1
    ```

3.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

    These canned models will be downloaded and extracted to a new folder
    ```all_models```.

4.  Clone this project:

    ```
    cd ~/google-coral

    git clone https://github.com/unixabg/object-identify-coral-tpu.git
    ```

4.  Enther the cloned project and run with sample mp4 to look for an object like dog:

    ```
    cd ~/google-coral/object-identify-coral-tpu

    python3 detect.py --model ../examples-camera/all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels ../examples-camera/all_models/coco_labels.txt --video_file dog.mp4 --output_dir ./detected_frames --detect_object dog
    ```

