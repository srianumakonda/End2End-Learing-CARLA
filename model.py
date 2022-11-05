import torch
import torchvision
import numpy as np
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()

        self.model = nn.Sequential(
            self._block(3, 16, 3),
            self._block(16, 32, 3),
            self._block(32, 64, 3),
            self._block(64, 128, 3),
            nn.Conv2d(128, 1, 1, stride=[4,3]),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # self._init_weights()

    def forward(self, x):
        x = self.model(x).view(-1)
        x[0] = torch.tanh(x[0]) #we take the domain to be between [-1, 1] but we can then multiply it by 70 to know the min and max steering values (carla only accepts [-1, 1] as the domain for the steering control)
        x[1] = torch.sigmoid(x[1])
        x[2] = torch.sigmoid(x[2])
        x[3] = torch.sigmoid(x[3])
        return x

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

    def _block(self, channel_in, channel_out, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size, stride=stride),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, kernel_size, stride=stride),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )


if __name__ == "__main__":
    steering_model = SteeringModel().to("cuda")

    y = steering_model(torch.randn((1,3,128,256),requires_grad=False).to("cuda"))
    print(y.view(-1))
    # with torch.no_grad():
    #     print(i for i in x)
    # summary(steering_model, (3, 128, 256))