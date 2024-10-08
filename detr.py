import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    image_path = "/home/aistation/Downloads/pose.jpg"
    image = Image.open(image_path)

    image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("models/deformable_detr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        0
    ]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

# Load Pre-trained MIM model (DensePose + MIM)
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "path/to/pretrained_mim_model.pth"  # Replace with the correct path
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only person class
# mim_predictor = DefaultPredictor(cfg)
#
# # Define the transform
# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
# ])
#
# def detect_athletes(image_path):
#     """
#     Detect two athletes on the tatami using Deformable DETR and MIM to filter out background.
#     """
#     # Load the image and apply transformations
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0)
#
#     # Step 1: Detect all persons using Deformable DETR
#     with torch.no_grad():
#         outputs = model(image_tensor)
#     boxes = outputs['pred_boxes']  # Bounding boxes
#     labels = outputs['pred_logits'].argmax(-1)  # Predicted class labels
#
#     # Step 2: Filter only person class (assumed as class 1 in COCO)
#     person_boxes = boxes[labels == 1]
#
#     # Step 3: Apply MIM using DensePose model to suppress background
#     predictions = mim_predictor(image)
#     instances = predictions['instances'].to("cpu")
#
#     # Filter out only 2 athletes using some post-processing based on bounding box sizes or positions
#     person_scores = instances.scores
#     person_boxes = instances.pred_boxes
#     top_boxes = person_boxes[person_scores.argsort(descending=True)[:2]]  # Keep only top 2 persons
#
#     return top_boxes
#
# # Test detection on an example image
# image_path = 'path_to_image.jpg'
# athlete_boxes = detect_athletes(image_path)
# print("Detected Athlete Bounding Boxes:", athlete_boxes)
