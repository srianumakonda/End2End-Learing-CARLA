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
            nn.Flatten(),
            # nn.Dropout(0.3),
            nn.Linear(6144, 512),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(512,3),
        )

        self.steer = nn.Linear(512,1)
        self.other = nn.Linear(512,2)
        self.out = nn.Linear(512,3)

        self._init_weights()

    def forward(self, x):
        x = self.model(x)
        out = self.out(x)
        # steer = torch.tanh(self.steer(x))
        # other = torch.sigmoid(self.other(x))
        # out = torch.tensor(torch.cat((steer,other),dim=1).requires_grad_(True),requires_grad=True)
        return out

        # print(x[:,0])

        # steer = torch.tanh(x[:,0]) #we take the domain to be between [-1, 1] but we can then multiply it by 70 to know the min and max steering values (carla only accepts [-1, 1] as the domain for the steering control)
        # throttle = torch.sigmoid(x[:,1])
        # brake = torch.sigmoid(x[:,2])
        # # print(steer)
        # # reverse = torch.sigmoid(x[3])
        # return torch.tensor((steer, throttle, brake), requires_grad=True)#brake, reverse], requires_grad=True)

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
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, kernel_size, stride=stride),
            nn.BatchNorm2d(channel_out),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )


if __name__ == "__main__":
    steering_model = SteeringModel().to("cuda")

    y = steering_model(torch.randn((64,3,128,256),requires_grad=False).to("cuda"))
    print(y)
    for i in y[:,1]:
        if i<0:
            print("False")
    for i in y[:,2]:
        if i<0:
            print("False")
    # with torch.no_grad():
    #     print(i for i in x)
    # summary(steering_model, (3, 128, 256))