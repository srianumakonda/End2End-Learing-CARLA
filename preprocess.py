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
        self.vals = [line for line in open("output/data.txt")][1:]

        self.steer = []
        self.throttle = []
        self.brake = []
        self.reverse = []

        for i in self.vals:
            self.steer.append(float(i.split()[0]))
            self.throttle.append(float(i.split()[1]))
            self.brake.append(float(i.split()[2])) #it's 0.0 throughout but ill just leave it like that
            self.reverse.append(i.split()[3])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image[idx]),cv2.COLOR_BGR2RGB)/255.0
        data = self.vals[idx]

        steer = self.steer[idx]
        throttle = self.throttle[idx]
        brake = self.brake[idx]
        reverse = self.reverse[idx]

        if self.transform:
            img = self.transform(img)
        return img, [steer, throttle, brake, reverse]