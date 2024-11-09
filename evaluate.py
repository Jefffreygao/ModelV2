import torch
from torch.utils.data import DataLoader
from model import TumorDetectionModel
from dataset import TumorDataset
import numpy as np

test_data_path = 'data_splits/test_data'
model = TumorDetectionModel()
model.load_state_dict(torch.load('modelXY.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Load test dataset
test_dataset = TumorDataset(test_data_path, mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize metrics
correct_tumor_preds = 0
total_tumor_samples = 0
coord_errors = []
max_samples = 1000  # Limit to 1000 samples for evaluation for now


with torch.no_grad():
    for i, (images, masks, tumor_present, tumor_coords) in enumerate(test_loader):
        if i >= max_samples:  # Stop after 1000 images
            break

        images, tumor_present, tumor_coords = images.to(device), tumor_present.to(device), tumor_coords.to(device)

        tumor_pred, coords_pred = model(images)

        # Tumor presence prediction (binary classification)
        tumor_pred_class = (torch.sigmoid(tumor_pred) > 0.5).float()
        if tumor_pred_class == tumor_present:
            correct_tumor_preds += 1
        total_tumor_samples += 1

        # Coordinate error calculation if tumor is present
        if tumor_present.item() == 1:
            error = torch.sqrt(((coords_pred - tumor_coords) ** 2).sum()).item()
            coord_errors.append(error)


accuracy = correct_tumor_preds / total_tumor_samples * 100
avg_coord_error = np.mean(coord_errors) if coord_errors else 0.0

print(f"Tumor Presence Accuracy: {accuracy:.2f}%")
print(f"Average Coordinate Error (for detected tumors): {avg_coord_error:.2f} pixels")