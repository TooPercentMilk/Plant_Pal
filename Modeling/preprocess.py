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
    def __init__(self, filenames, ar_limits=(0, 100)):
        self.root_dir = filenames
        self.ar_limits = ar_limits
        self.image_paths = [os.path.join('../Data', fname) for fname in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_paths[idx].split('/')[:-1]
        label = int(self.image_paths[idx].split('/')[-1])
        image_path = self.image_paths[idx]
        image = resize_and_sharpen(image_path, self.ar_limits)
        return image, label
    #TODO: handle None images returned from resize_and_sharpen

class Historian():
    def __init__(self, early_stopping=2):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.stop = early_stopping
    
    def record(self, loss, accuracy, val_loss, val_accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        if len(self.val_losses) < self.stop:
            return True
        for i in range(self.stop):
            if self.val_losses[-1 - i] > self.val_losses[-2 - i]:
                return False
        return True
    
    def plot(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(self.losses, label='train loss')
        ax[0].plot(self.val_losses, label='val loss')
        ax[0].set_title('Loss')
        ax[0].legend()
        ax[1].plot(self.accuracies, label='train accuracy')
        ax[1].plot(self.val_accuracies, label='val accuracy')
        ax[1].set_title('Accuracy')
        ax[1].legend()
        plt.show()