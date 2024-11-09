import os
import json

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import transforms
from argparse import ArgumentParser

from classifier.position_classifier import PositionClassifier


# Mapping of class index to BJJ positions
label_to_position = {
    0: 'standing',
    1: 'takedown1',
    2: 'takedown2',
    3: 'open_guard1',
    4: 'open_guard2',
    5: 'half_guard1',
    6: 'half_guard2',
    7: 'closed_guard1',
    8: 'closed_guard2',
    9: '5050_guard',
    10: 'side_control1',
    11: 'side_control2',
    12: 'mount1',
    13: 'mount2',
    14: 'back1',
    15: 'back2',
    16: 'turtle1',
    17: 'turtle2'
}


def parse_args():
    parser = ArgumentParser(description='Test the position classifier model')
    parser.add_argument('--model', type=str, default='position_classifier.pth', help='Path to the model file')
    parser.add_argument('--image', type=str, default='0000040.jpg', help='Path to the image file')
    parser.add_argument('--keypoints', type=str, default='annotations.json', help='Path to the annotations file')
    return parser.parse_args()


# Function to display the image with the predicted label using PIL
def display_image_with_label(image_resized, predicted_label):
    plt.imshow(image_resized)
    plt.axis('off')  # Hide axes
    plt.title('Predicted Position: ' + predicted_label)
    plt.show()


def rescale_keypoints(keypoints, original_width, original_height, target_size=(256, 256)):
    """Rescale the keypoints to the target image size."""
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height
    rescaled_keypoints = []
    for kp in keypoints:
        x_rescaled = kp[0] * scale_x
        y_rescaled = kp[1] * scale_y
        rescaled_keypoints.append([x_rescaled, y_rescaled])
    return rescaled_keypoints


def get_keypoints_by_image_id(keypoints_file, image_path):
    with open(keypoints_file, 'r') as f:
        annotations = json.load(f)

    if annotations is None:
        return [], []

    keypoints1 = []
    keypoints2 = []
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    for item in annotations:
        if item['image'] == image_name:
            # The poses of the athletes are in MS-COCO format
            pose1 = item.get('pose1', [[0, 0]] * 17)
            pose2 = item.get('pose2', [[0, 0]] * 17)

            # Use only x,y coordinates, skip confidence detector
            keypoints1.append([kp[:2] for kp in pose1])
            keypoints2.append([kp[:2] for kp in pose2])

            return keypoints1, keypoints2

    return [], []


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    num_joints, num_classes = 34, 18
    model = PositionClassifier(num_joints, num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))  # Load the model
    model.eval()  # Set to evaluation mode

    # Preprocessing function for images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.42640511379656704], std=[0.2448158246877738])  # Normalize image
    ])

    # Load the image and keypoints
    image = Image.open(args.image).convert('RGB')

    # Get original dimensions from the image
    original_width, original_height = image.size

    # Resize image to the target size
    image_tensor = transform(image).unsqueeze(0)

    keypoints1, keypoints2 = get_keypoints_by_image_id(args.keypoints, args.image)
    # Rescale keypoints to match the new image size (256x256)
    keypoints1 = rescale_keypoints(keypoints1[0], original_width, original_height)
    keypoints2 = rescale_keypoints(keypoints2[0], original_width, original_height)

    # Flatten coordinates to 1D list
    keypoints1_flat = [coord for kp in keypoints1 for coord in kp]
    keypoints2_flat = [coord for kp in keypoints2 for coord in kp]

    keypoints_tensor = torch.tensor(np.array(keypoints1_flat + keypoints2_flat), dtype=torch.float32).unsqueeze(0)

    # Pass image and keypoints through model
    with torch.no_grad():
        output = model(image_tensor, keypoints_tensor)
        _, predicted = torch.max(output, 1)  # Get the predicted label
        predicted_label = label_to_position[predicted.item()]

    # Display the image with the predicted label
    display_image_with_label(image, predicted_label)


if __name__ == '__main__':
    # Check Pytorch installation
    print('torch version:', torch.__version__, torch.cuda.is_available())
    print('torchvision version:', torchvision.__version__)

    main()
