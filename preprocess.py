import torch
import torchvision
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class CARLAPreprocess(Dataset):

    def __init__(self, transform=None):
        super(CARLAPreprocess, self).__init__()
        self.transform = transform
        self.image = glob.glob("output/*.jpg")

        self.image = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.reverse = []

        with open("output/data.txt") as f:
            for i, line in enumerate(f):
                self.image.append(cv2.cvtColor(cv2.imread(self.image[i]),cv2.COLOR_BGR2RGB)/255.0)
                self.steer.append(float(line.split()[0]))
                self.throttle.append(float(line.split()[1]))
                self.brake.append(float(line.split()[2])) #it's 0.0 throughout but ill leave it like that
                self.reverse.append(line.split()[3])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        steer = self.steer[idx]
        throttle = self.throttle[idx]
        brake = self.brake[idx]
        reverse = self.reverse[idx]

        if self.transform:
            img = self.transform(img)
        return img, [steer, throttle, brake, reverse]