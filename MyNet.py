import torch
from torch import nn

'''自定义VGG网络，但是由于电脑GPU内存过小导致程序无法运行'''
class VGG16(nn.Module):
    def __init__(self, nc):
        super(VGG16, self).__init__()
        self.features = nn.Sequential()
        in_channel = 3
        layers = [2, 2, 3, 3, 3]
        channels = [64, 128, 256, 512, 512]

        for i, n in enumerate(layers):
            for j in range(n):
                self.features.add_module("conv{}_{}".format(i, j),nn.Conv2d(
                    in_channel, channels[i], 3, stride=1, padding=1
                ))
                self.features.add_module("relu{}_{}".format(i, j), nn.ReLU(inplace=True))
                in_channel = channels[i]
            self.features.add_module("pool_{}".format(i), nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nc)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class VGG11(nn.Module):
    def __init__(self, nc):
        super(VGG11, self).__init__()
        self.features = nn.Sequential()
        in_channel = 3
        layers = [1, 1, 2, 2, 2]
        channels = [64, 128, 256, 512, 512]
        for i, n in enumerate(layers):
            for j in range(n):
                self.features.add_module("conv{}_{}".format(i, j), nn.Conv2d(in_channel, channels[i], 3, stride=1, padding=1))
                self.features.add_module("relu{}_{}".format(i, j), nn.ReLU(inplace=True))
                in_channel = channels[i]
            self.features.add_module("pool{}".format(i), nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, nc)
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x




