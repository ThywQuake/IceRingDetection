import torch
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights

# Custom Backbone for 7-band images
class CustomBackbone(torch.nn.Module):
    def __init__(self, original_model):
        super(CustomBackbone, self).__init__()
        # Modify the first convolutional layer of the original model
        self.features = torch.nn.Sequential(
            # Adjust the input channels of the first layer to 7
            torch.nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(original_model.children())[1:]
        )

    def forward(self, x):
        return self.features(x)

def create_model(num_classes):
    # Load a pre-trained model for the backbone using the new 'weights' argument
    weights = ResNet50_Weights.DEFAULT
    original_model = models.resnet50(weights=weights)

    # Replace the first layer with our custom layer
    custom_backbone = CustomBackbone(original_model)

    # Anchor Generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone=custom_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # Additional parameters like box_roi_pool can be set here
    )
    return model


model=create_model(num_classes=2)
