import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if '_Full.npy' in f])
        self.mode = mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image alongside its corresponding mask file
        image_file = self.image_files[idx]
        mask_file = image_file.replace('_Full.npy', '_Mask.npy')

        image_path = os.path.join(self.data_dir, image_file)
        mask_path = os.path.join(self.data_dir, mask_file)

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        # Convert to tensor
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        # see if there is tumor present
        tumor_present = 1 if mask.sum() > 0 else 0
        tumor_present = torch.tensor([tumor_present], dtype=torch.float32)
        #tumor coordinate stuff
        max_tumors = 10
        tumor_coords = torch.zeros((0, 2), dtype=torch.float32)  # Initialize as empty 2D tensor
        if tumor_present:
            # Identify all tumor regions
            tumor_pixels = torch.nonzero(mask[0], as_tuple=False)
            if len(tumor_pixels) > 0:
                tumor_coords = tumor_pixels.float()  # Update with actual coordinates

        if tumor_coords.size(0) < max_tumors:
            tumor_coords = F.pad(tumor_coords, (0, 0, 0, max_tumors - tumor_coords.size(0)), "constant", 0)
        else:
            tumor_coords = tumor_coords[:max_tumors]

        return image, mask, tumor_present, tumor_coords
