import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import timm


class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()

        # Load ResNet50 as baseline model
        weights = ResNet50_Weights.DEFAULT
        self.resnet50 = resnet50(weights=weights)

        # Replace final layer
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)


# Patch4 / Patch16 = patch size of 4x4 or 16x16 pixels
# Smaller size = more patches -> finer spatial granularity but increase sequence length
# 224 = images are resized to 224x224 pixels before patch splitting
class ViTModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTModel, self).__init__()

        # Create ViT base model
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

        # Replace classification head 
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)