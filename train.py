import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import TumorDetectionModel
from dataset import TumorDataset


train_data_path = 'data_splits/train_data'
validation_data_path = 'data_splits/validation_data'
learning_rate = 0.001
batch_size = 16
epochs = 20

model = TumorDetectionModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

classification_loss_fn = nn.BCEWithLogitsLoss()  # Binary loss for tumor presence
coordinate_loss_fn = nn.MSELoss()  # Mean squared error for tumor location
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load datasets and create data loaders to manage batching
train_dataset = TumorDataset(train_data_path, mode='train')
validation_dataset = TumorDataset(validation_data_path, mode='validation')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Start training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, masks, tumor_present, tumor_coords in train_loader:
        images, tumor_present, tumor_coords = images.to(device), tumor_present.to(device), tumor_coords.to(device)

        optimizer.zero_grad()
        # Forward pass
        tumor_pred, coords_pred = model(images)

        loss_classification = classification_loss_fn(tumor_pred, tumor_present)

        coord_mask = tumor_present.bool()
        loss_coords = coordinate_loss_fn(coords_pred[coord_mask], tumor_coords[coord_mask])

        # Combine classification and coordinate losses, backpropagate, and update model
        loss = loss_classification + loss_coords
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  #total loss

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}")


# Save the model
torch.save(model.state_dict(), 'modelXY.pth')
print("Training complete, model saved as 'modelXY.pth'")