import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt

dir = "../FLIC/images/"
csv_path = "../FLIC.csv"
checkpoint_dir = "../checkpoint/"
img_size = (200, 200)
full_size = (480, 720)
num_outputs = 22
batch_size = 32

df = pd.read_csv(csv_path)
df = df.drop(columns="Unnamed: 23")


def plot_points(coords):
    x = [coords[i] for i in range(len(coords)) if i % 2 == 0]
    y = [coords[i] for i in range(len(coords)) if i % 2 == 1]
    plt.plot(x, y, 'bo', markersize=3)

class KeypointsDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['img_name']
        img_path = os.path.join(self.img_dir, img_name)

        # Read image with OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # Channels first

        keypoints = self.df.iloc[idx, 1:].values.astype(np.float32)

        return torch.tensor(img), torch.tensor(keypoints)


dataset = KeypointsDataset(df, dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



class KeypointModel(nn.Module):  # or I'll put my HW-3 one here
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointModel().to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_checkpoint(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

def train_model(model, dataloader, epochs=30, steps_per_epoch=5, checkpoint_path_in=None):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (imgs, keypoints) in enumerate(dataloader):
            if i >= steps_per_epoch:
                break

            imgs, keypoints = imgs.to(device), keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / steps_per_epoch:.4f}")
        if checkpoint_path_in:
            save_checkpoint(model, checkpoint_path_in)


checkpoint_path = os.path.join(checkpoint_dir, "cp.pth")
train_model(model, dataloader, epochs=30, steps_per_epoch=5, checkpoint_path_in=checkpoint_path)


def test_model(model):
    model.eval()
    rand_idx = random.randint(0, len(df) - 1)
    img_name = df.iloc[rand_idx]["img_name"]
    img_path = os.path.join(dir, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = torch.tensor(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        coords = model(img_input).cpu().numpy().flatten()

    img_display = cv2.resize(img, full_size)
    plt.imshow()


test_model(model)

load_checkpoint(model, checkpoint_path)
checkpoint_path2 = os.path.join(checkpoint_dir, "cp2.pth")
train_model(model, dataloader, epochs=10, steps_per_epoch=15, checkpoint_path_in=checkpoint_path2)
