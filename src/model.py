import torch.nn as nn
import torchvision.models as models


class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50Model, self).__init__()

        # Load ResNet50 as baseline model
        self.resnet50 = models.resnet50(pretrained=pretrained)

        # Replace final layer
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
