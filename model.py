import torch
import torchvision
import numpy as np
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()


        # INPUT SHAPE: 
        # Recreating NVIDIAs network
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*1*18, 1164), 
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(1164, 100),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(100, 50),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(50, 10),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(10, 1),
        )

        self.conv1 = nn.Conv2d(3, 24, 5, 2)
        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64*1*18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 4)

        # self._init_weights()

    def forward(self, x):
        x = self.model(x)
        return torch.tanh(x) #we take the domain to be between [-1, 1] but we can then multiply it by 70 to know the min and max steering values (carla only accepts [-1, 1] as the domain for the steering control)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)


