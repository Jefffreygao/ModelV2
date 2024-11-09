import torch
import torch.nn as nn
import torch.nn.functional as F


class TumorDetectionModel(nn.Module):
    def __init__(self, max_tumors=10):
        super(TumorDetectionModel, self).__init__()
        # Convolutional layers for feature extraction
        self.max_tumors = max_tumors
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for classification and regression
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc_classification = nn.Linear(128, 1)  # Binary classification output (tumor presence)
        self.fc_coordinates = nn.Linear(128, max_tumors * 2)  # Predict up to 10 (x, y) pairs

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        #classification and coordinates
        tumor_presence = self.fc_classification(x)
        tumor_coords = self.fc_coordinates(x).view(-1, self.max_tumors, 2)

        return tumor_presence, tumor_coords
