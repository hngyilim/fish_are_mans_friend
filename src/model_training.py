import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim


def load_pretrained_model(device, model_name, classes):
    """
    Loads pretrain model, freezes parameters and adds a final layer for transfer learning and ensuring output number of classes is the same
    :param device: torch.device('cpu') or torch.device('gpu')
    :param model_name: str ('available models are vgg16 ... )
    :param classes: target classes 

    """
    # Basic Model
    if model_name == 'vgg16':
        model_ft = models.vgg16(pretrained=True)
        # Freeze model parameters
        for param in model_ft.parameters():
            param.requires_grad = False

        # Change the final layer of VGG16 Model for Transfer Learning
        # Here the size of each output sample is set to 5
        fc_inputs = model_ft.classifier[-4].out_features
        model_ft.classifier[-1] = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(classes))
        )

        # Enable GPU usage for model weights
        model_ft = model_ft.to(device)
        return model_ft
        
        