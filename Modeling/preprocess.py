import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# determined in EDA notebook
channel_means = [0.45587662, 0.49715131, 0.37855863]
channel_stds = [0.18720555, 0.17101169, 0.19548449]
# Adjust to whatever pretrained model expects
res_means = [0.485, 0.456, 0.406]
res_stds = [0.229, 0.224, 0.225]

def resize_and_sharpen(image_path, ar_limits=(0, 100)):
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio < ar_limits[0] or aspect_ratio > ar_limits[1]:
            return None # I have decided to ignore files that will get too distorted by the resizing
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda img: transforms.functional.adjust_sharpness(img, sharpness_factor=2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=res_means, std=res_stds)
        ])
        img_transformed = transform(img)
        return img_transformed
# TODO: find optimal sharpness factor and ar_limits


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ar_limits=(0, 100)):
        self.root_dir = data_dir
        self.ar_limits = ar_limits
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(('jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = resize_and_sharpen(image_path, self.ar_limits)
        return image
    #TODO: handle None images returned from resize_and_sharpen