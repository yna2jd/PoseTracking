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

import mediapipe as mp
from testing_mat import *

dir = "FLIC/images/"
csv_path = "FLIC.csv"
checkpoint_dir = "checkpoint/"
img_size = (200, 200)
# img_size = (480, 720)
full_size = (480, 720)
num_outputs = 22
batch_size = 32

df = pd.read_csv(csv_path)
df = df.drop(columns="Unnamed: 23")

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks

def plot_points(coords):
    x = [coords[i] for i in range(len(coords)) if i % 2 == 0]
    y = [coords[i] for i in range(len(coords)) if i % 2 == 1]
    plt.plot(x, y, 'ro', markersize=3)

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
        # img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))  # Channels first

        keypoints = self.df.iloc[idx, 1:].values.astype(np.float32)

        return torch.tensor(img), torch.tensor(keypoints)


dataset = KeypointsDataset(df, dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



class KeypointModel(nn.Module):  # or I'll put my HW-3 one here
    def __init__(self):
        super(KeypointModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28 -> 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 14 -> 7

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 7 -> 3

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 3 -> 1
            # nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.14),
            nn.Flatten(),
            # nn.Linear(32, 10)
            nn.Linear(36864, 22)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
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


checkpoint_path = os.path.join(checkpoint_dir, "cp4.pth")
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
    plt.imshow(img_display)
    plt.axis("off")
    
    # our model
    print(coords)
    plot_points(coords)
    plt.show()

    # actual answer
    label_points(rand_idx)
    plt.show()

    # mediapipe
    results = pose.process(img_display)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img_display, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on temp
    cv2.imshow(img_name, img_display)
    cv2.waitKey(0)

load_checkpoint(model, checkpoint_path)
checkpoint_path2 = os.path.join(checkpoint_dir, "cp5.pth")
train_model(model, dataloader, epochs=10, steps_per_epoch=15, checkpoint_path_in=checkpoint_path2)

test_model(model)