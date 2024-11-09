# `original_keypoints_check.py` Utility

## Overview

The `original_keypoints_check.py` script is designed to test a position classifier model for Brazilian Jiu-Jitsu (BJJ) positions. It processes an input image and its corresponding keypoints, passes them through a pre-trained model, and displays the image with the predicted position label.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install torch torchvision numpy matplotlib pillow
    ```

## Usage

To run the script, use the following command:
```sh
python utils/original_keypoints_check.py --model <path_to_model> --image <path_to_image> --keypoints <path_to_keypoints>
```

### Arguments
* --model: Path to the model file (default: position_classifier.pth).
* --image: Path to the image file (default: 0000040.jpg).
* --keypoints: Path to the annotations file (default: annotations.json).

### Example
```sh
python utils/original_keypoints_check.py --model models/position_classifier.pth --image data/images/0000040.jpg --keypoints data/annotations/annotations.json
```

## Script Details

### Functions
* parse_args(): Parses command-line arguments.
* display_image_with_label(image_resized, predicted_label): Displays the image with the predicted label using matplotlib.
* rescale_keypoints(keypoints, original_width, original_height, target_size): Rescales the keypoints to the target image size.
* get_keypoints_by_image_id(keypoints_file, image_path): Retrieves keypoints from the annotations file based on the image ID.
* main(): Main function that loads the model, processes the image and keypoints, and displays the predicted label.

### Key Points
* The script uses a pre-trained PositionClassifier model to predict BJJ positions.
* The input image is resized to 256x256 pixels.
* Keypoints are rescaled to match the resized image dimensions.
* The predicted position is displayed on the image using matplotlib.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for more details.
