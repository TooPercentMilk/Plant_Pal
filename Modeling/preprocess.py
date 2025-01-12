import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
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

        self.classes = os.listdir('../Data')
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        self.encoded_labels = self.label_encoder.transform(self.classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_paths[idx].split('/')[2]
        label = self.label_encoder.transform([label])[0]
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
        self.best_model_state = None
        self.best_optimizer_state = None
        self.stop = early_stopping
    
    def record(self, loss, accuracy, val_loss, val_accuracy, model_state, opt_state):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        if val_loss < min(self.val_losses):
            self.best_model_state = model_state
            self.best_optimizer_state = opt_state
        if len(self.val_losses) < self.stop:
            return True
        for i in range(self.stop):
            if self.val_losses[-1 - i] > self.val_losses[-2 - i]:
                print('Early stopping reached - terminating training loop')
                return False
        return True
    
    def save_model(self):
        best_ind = np.argmin(self.val_losses)
        checkpoint = {
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.best_optimizer_state,
            'loss': self.val_losses[best_ind],
            'accuracy': self.val_accuracies[best_ind]
        }
        torch.save(checkpoint, '../Best_Models/best_model.pth')
    
    def performance(self, verbose=False):
        if verbose:
            print(f'Training loss: {self.losses[-1]}, Training accuracy: {self.accuracies[-1]}')
            print(f'Validation loss: {self.val_losses[-1]}, Validation accuracy: {self.val_accuracies[-1]}')
        return self.losses[-1], self.accuracies[-1], self.val_losses[-1], self.val_accuracies[-1]
    
    def final_performance(self, verbose=False):
        best_ind = np.argmin(self.val_losses)
        print('Training Finished - Best Model Performance:')
        print(f'Validation loss = {self.val_losses[best_ind]}, Validation accuracy = {self.val_accuracies[best_ind]}')
        if verbose:
            self.plot()
        return self.losses[best_ind], self.accuracies[best_ind], self.val_losses[best_ind], self.val_accuracies[best_ind]
    
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