import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import mmcv

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from tqdm import tqdm

import mmpose
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import os.path as osp
import cv2
import hashlib

from classifier.position_classifier import PositionClassifier


def get_image_name(image, file_name):
    # Generate hash (you can hash the file contents or any string)
    hash_string = hashlib.md5(image).hexdigest()[:8]  # Getting first 8 characters of the hash

    # Append hash to the file name
    name, ext = osp.splitext(file_name)
    new_file_name = f"{name}_{hash_string}{ext}"
    return new_file_name


# Frame extraction from video
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            frames.append(frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames


# Preprocessing function for image frames
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


def best_keypoints_scores(keypoints, keypoint_scores):
    # Sum the keypoint scores to get an overall confidence score per athlete
    overall_scores = keypoint_scores.sum(axis=1)  # Shape: (num_people,)

    # Get the indices of the top 2 athletes by overall keypoint score
    top_2_indices = np.argsort(overall_scores)[-2:]  # Take the last two (highest scores)

    # Select the keypoints for the top 2 athletes
    top_2_keypoints = keypoints[top_2_indices]  # Shape: (2, 17, 2)

    # Select the keypoint scores for the top 2 athletes
    top_2_keypoint_scores = keypoint_scores[top_2_indices]  # Shape: (2, 17)

    return top_2_keypoints, top_2_keypoint_scores


def extract_keypoints(image, detector, pose_estimator):
    # Convert the input image to the format that MMPose expects
    img = mmcv.imread(image)

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    # Check if pose_results is empty to avoid index errors
    if not pose_results:
        return True, torch.zeros(68)  # Return zero tensor if less than 2 keypoints are detected

    # Extract keypoints and scores from all detected people
    keypoints_list = []
    keypoint_scores_list = []
    for result in pose_results:
        if hasattr(result, 'pred_instances'):
            if result.pred_instances is None:
                continue
            pred_instances = result.pred_instances
            keypoints = pred_instances.keypoints  # (N, 17, 2) for detected person's keypoints
            keypoint_scores = pred_instances.keypoint_scores  # (N, 17) for confidence scores

            keypoints_list.append(keypoints)
            keypoint_scores_list.append(keypoint_scores)

    # Concatenate the keypoints and scores for all people
    keypoints_all = np.concatenate(keypoints_list, axis=0)  # Shape: (num_people, 17, 2)
    keypoint_scores_all = np.concatenate(keypoint_scores_list, axis=0)  # Shape: (num_people, 17)

    # If more than two people detected, filter for the top 2 by scores
    keypoints_top2, keypoint_scores_top2 = best_keypoints_scores(keypoints_all, keypoint_scores_all)

    keypoints_top2 = keypoints_top2.tolist()

    # Check if there are 34 keypoints (17 per person for two people)
    if len(keypoints_top2) != 2:
        return True, None

    while len(keypoints_top2) < 2:
        keypoints_top2.append([[0, 0]] * 17)

    # Flatten the keypoints for model input
    flattened_keypoints = torch.tensor(np.array(keypoints_top2).flatten(), dtype=torch.float32)
    return False, flattened_keypoints


# Analyze frames and predict positions using the model
def analyze_video(model, video_path, detector, pose_estimator):
    predictions = []

    frames = extract_frames(video_path, frame_rate=10)  # Extract every 10th frame for analysis
    for frame in tqdm(frames, desc="Processing frames"):
        # Convert frame to RGB (OpenCV loads images in BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to 256x256
        frame_resized = cv2.resize(frame_rgb, (256, 256))

        # Extract keypoints
        skip, keypoints = extract_keypoints(frame_resized, detector, pose_estimator)
        if skip:
            continue
        keypoints = keypoints.unsqueeze(0)  # Add batch dimension

        # Preprocess frame
        frame_tensor = preprocess(frame_resized).unsqueeze(0)  # Add batch dimension

        # Pass frame and keypoints through model
        with torch.no_grad():
            model.eval()
            output = model(frame_tensor, keypoints)
            _, predicted = torch.max(output, 1)  # Get the predicted label
            predictions.append(predicted.item())

    return predictions


# Visualize predictions
def visualize_predictions(video_path, predictions, label_to_position):
    if len(predictions) == 0:
        print("No predictions to visualize.")
        return

    frames = extract_frames(video_path, frame_rate=10)
    rows, cols = 10, 4  # Display 40 frames as a grid

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(min(len(frames), 40)):
        ax = fig.add_subplot(rows, cols, i + 1)
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        ax.imshow(frame)
        ax.set_title(label_to_position[predictions[i]])  # Display predicted position
        ax.axis('off')

    plt.tight_layout()
    plt.show()


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize MMPose RTMPose model once
    pose_config = 'configs/rtmpose/estimator/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    det_config = 'configs/rtmpose/detector/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    # build detector
    detector = init_detector(
        det_config,
        det_checkpoint,
        device='cuda:0',
    )

    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device='cuda:0',
        cfg_options=cfg_options
    )

    # init visualizer
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    # Load the trained model
    num_joints = 34
    num_classes = 18  # Replace with the number of classes in your model
    model = PositionClassifier(num_joints, num_classes)
    model.load_state_dict(torch.load("position_classifier_v4.pth", map_location=device))  # Load the model

    # Analyze the video
    video_path = './classifier/assets/mp4/bjj_roll.mp4'  # Path to the video file
    predictions = analyze_video(model, video_path, detector, pose_estimator)

    # Visualize predictions
    visualize_predictions(video_path, predictions, label_to_position)


# Example usage
if __name__ == "__main__":
    # Check Pytorch installation
    print('torch version:', torch.__version__, torch.cuda.is_available())
    print('torchvision version:', torchvision.__version__)

    # Check MMPose installation
    print('mmpose version:', mmpose.__version__)

    # Check mmcv installation
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version

    print('cuda version:', get_compiling_cuda_version())
    print('compiler information:', get_compiler_version())

    try:
        from mmdet.apis import inference_detector, init_detector
        has_mmdet = True
    except (ImportError, ModuleNotFoundError):
        has_mmdet = False

    main()