import torch
import torch.nn as nn
import torchvision.models as models


class PositionClassifier(nn.Module):
    def __init__(self, num_joints, num_classes):
        super(PositionClassifier, self).__init__()

        # Image branch (CNN like ResNet)
        self.image_branch = models.resnet18(pretrained=True)
        self.image_branch.fc = nn.Linear(self.image_branch.fc.in_features, 512)

        # Annotation branch (MLP)
        self.annotation_branch = nn.Sequential(
            nn.Linear(num_joints * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Combined features
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, annotations):
        # Process image through the CNN
        image_features = self.image_branch(image)

        # Process annotations through the MLP
        annotation_features = self.annotation_branch(annotations)

        # Concatenate both feature vectors
        combined = torch.cat((image_features, annotation_features), dim=1)

        # Pass through final fully connected layers
        output = self.fc(combined)

        return output