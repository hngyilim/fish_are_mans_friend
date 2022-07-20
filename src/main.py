import streamlit as st
import torchvision
from torchvision import transforms, datasets
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

label_map = {
    0: "ALB",
    1: "BET",
    2: "DOL",
    3: "LAG",
    4: "NoF",
    5: "OTHER",
    6: "SHARK",
    7: "YFT",
}

label_list= [
    "ALB",
    "BET",
    "DOL",
    "LAG",
    "NoF",
    "OTHER",
    "SHARK",
    "YFT"
]

def predict_proba(
        model,
        img: Image.Image,
        k: int,
        index_to_class_labels: dict,
        show: bool = False
        ):
    """
    Feeds single image through network and returns
    top k predicted labels and probabilities

    params
    ---------------
    img - PIL Image - Single image to feed through model
    k - int - Number of top predictions to return
    index_to_class_labels - dict - Dictionary
        to map indices to class labels
    show - bool - Whether or not to
        display the image before prediction - default False

    returns
    ---------------
    formatted_predictions - list - List of top k
        formatted predictions formatted to include a tuple of
        1. predicted label, 2. predicted probability as str
    """
    
    model.eval()
    output_tensor = model(img)
    prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
    top_k = torch.topk(prob_tensor, k, dim=1)
    probabilites = top_k.values.detach().numpy().flatten()
    indices = top_k.indices.detach().numpy().flatten()
    formatted_predictions = []

    for pred_prob, pred_idx in zip(probabilites, indices):
        predicted_label = index_to_class_labels[pred_idx].title()
        predicted_perc = pred_prob * 100
        formatted_predictions.append(
            (predicted_label, f"{predicted_perc:.3f}%"))

    return formatted_predictions

@st.cache(allow_output_mutation=True)
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

with st.spinner('Model is being loaded..'):
    PATH = Path(__file__).resolve().parent.parent/'models'/'VGG16_v3_25_0.672.pt'
    # Use cuda to enable gpu usage for pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = torch.load(PATH,map_location=device)
    st.write(model_ft)


st.write("""
         # Fishy Classification
         """
         )

file = st.file_uploader("Please upload your dear fishy file", type=["jpg","png"])


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache()
def import_and_predict(image_data: Image.Image, model, k: int, index_to_label_dict: dict)-> list:
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

    actual_img = transform(image_data)
    actual_img = actual_img.unsqueeze(0) # add one dimension to the front to account for batch_size

    formatted_predictions = model(actual_img)

    st.write(formatted_predictions)

    # model.eval()
    # output_tensor = model(actual_img)
    # prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
    # top_k = torch.topk(prob_tensor, k, dim=1)
    # probabilites = top_k.values.detach().numpy().flatten()
    # indices = top_k.indices.detach().numpy().flatten()
    # formatted_predictions = []

    # for pred_prob, pred_idx in zip(probabilites, indices):
    #     predicted_label = label_map[pred_idx].title()
    #     predicted_perc = pred_prob * 100
    #     formatted_predictions.append(
    #         (predicted_label, f"{predicted_perc:.3f}%"))

    # return formatted_predictions

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    model_ft.eval()
    predictions = import_and_predict(image, model_ft, k = 3, index_to_label_dict = label_map)

    st.write(predictions[0][0])
 
    print(
    "This image most likely belongs to {}."
    .format(predictions[0][0])
)