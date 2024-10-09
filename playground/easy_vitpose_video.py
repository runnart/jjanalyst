import cv2
import tqdm
import time
from easy_ViTPose import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader

import numpy as np

# Paths to your models
model_path = '/home/aistation/mybjj/models/vitpose/vitpose_base-coco.pth'
yolo_path = '/home/aistation/mybjj/models/yolo/yolov8s.pt'
video_path = '/home/aistation/mybjj/classifier/assets/mp4/bjj_roll_3.mp4'  # Replace with the path to your video file


# Desired resolution for displaying the video
DISPLAY_WIDTH = 1280  # Desired width
DISPLAY_HEIGHT = 720   # Desired height


if __name__ == '__main__':
    # Initialize the model for video inference (set is_video=True)
    model = VitInference(model_path, yolo_path, model_name='s', yolo_size=320, is_video=True, device='cuda')

    reader = VideoReader(video_path)
    cap = cv2.VideoCapture(video_path)  # type: ignore
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    print(f'>>> Running inference on {video_path}')
    keypoints = []
    fps = []
    tot_time = 0.0
    paused = False
    current_frame = 0  # Track the current frame index

    # Initialize tqdm progress bar
    progress_bar = tqdm.tqdm(total=total_frames, desc="Processing Frames")

    reader = iter(reader)
    while current_frame < total_frames:
        if not paused:
            # Read the next frame from the video
            img = next(reader)
            if not img.any():
                print("Error: Could not read frame.")
                break

            t0 = time.time()

            # Run inference
            frame_keypoints = model.inference(img)
            keypoints.append(frame_keypoints)

            delta = time.time() - t0
            tot_time += delta
            fps.append(delta)

            # Draw the keypoints and YOLO detections
            img_display = model.draw(True, False)[..., ::-1]
            # Resize the image for display
            img_display_resized = cv2.resize(img_display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow('preview', img_display_resized)

            current_frame += 1  # Increment the current frame index
            progress_bar.update(1)  # Update progress bar with each processed frame
        else:
            # If paused, use the last processed frame for display
            if current_frame > 0:  # Ensure we have a frame to display
                cv2.imshow('preview', img_display_resized)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the video
            break
        elif key == ord(' '):  # Spacebar to pause/resume
            paused = not paused

    progress_bar.close()  # Close the progress bar

    tot_poses = sum(len(k) for k in keypoints)
    print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
    print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
          f'{(tot_poses / current_frame):.2f}')
    print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

    cv2.destroyAllWindows()

    # Reset the model after video inference
    model.reset()
