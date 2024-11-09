import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import TumorDetectionModel
from dataset import TumorDataset
import os

train_data_path = 'data_splits/train_data'
validation_data_path = 'data_splits/validation_data'

learning_rate = 0.001
batch_size = 16
epochs = 20
max_tumors = 10

# Initialize the model and move it to GPU if available
model = TumorDetectionModel(max_tumors=max_tumors)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up loss functions
classification_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss for tumor presence
coordinate_loss_fn = nn.MSELoss()  # Mean squared error for tumor coordinates

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TumorDataset(train_data_path, mode='train')
validation_dataset = TumorDataset(validation_data_path, mode='validation')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

#training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    processed_images = 0

    for images, masks, tumor_present, tumor_coords in train_loader:
        images, tumor_present, tumor_coords = images.to(device), tumor_present.to(device), tumor_coords.to(device)

        optimizer.zero_grad()
        tumor_pred, coords_pred = model(images)
        loss_classification = classification_loss_fn(tumor_pred, tumor_present)

        # Calculate coordinate loss only for images where tumors
        coord_mask = tumor_present.bool().unsqueeze(-1).expand(-1, coords_pred.size(1), coords_pred.size(2))
        loss_coords = coordinate_loss_fn(coords_pred[coord_mask], tumor_coords[coord_mask])

        loss = loss_classification + loss_coords

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        processed_images += images.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Processed {processed_images}/{len(train_loader.dataset)} images", end="\r")

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Save the trained model to a file
torch.save(model.state_dict(), 'modelXY.pth')
print("Model training complete and saved as 'modelXY.pth'")
