
from torchvision import transforms, datasets

def transform(model_name='vgg16'):
    if 'vgg' in model_name:
        transform = {'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                        ),
                    'val': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                        )
                    }
    if 'resnet' in model_name  or 'alexnet' in model_name or 'ResNeXt' in model_name:
        transform =  {'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                    ),
                    'val':
                        transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                        )
                    }
    if 'inception' in model_name:
        transform =  {'train': transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                    ),
                    'val':
                        transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    }
            
    return transform