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
import pandas as pd

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

predicted_to_actual_dict = {
    "ALB" : 'Albacore Tuna',
    "BET" : 'Bigeye Tuna',
    "DOL" : 'Dolphinfish, Mahi Mahi',
    "LAG" : 'Opah, Moonfish',
    "NoF" : 'No Fish',
    "OTHER" : 'Fish present but not in target categories',
    "SHARK" : 'Shark, including Silky & Shortfin Mako',
    "YFT" : 'Yellowfin Tuna'
    }

fish_to_wiki  = {
    0: "https://en.wikipedia.org/wiki/Albacore",
    1: "https://en.wikipedia.org/wiki/Bigeye_tuna",
    2: "https://en.wikipedia.org/wiki/Mahi-mahi",
    3: "https://en.wikipedia.org/wiki/Opah",
    4: "https://en.wikipedia.org/wiki/Fish",
    5: "https://en.wikipedia.org/wiki/Fish",
    6: "https://en.wikipedia.org/wiki/Shark",
    7: "https://en.wikipedia.org/wiki/Yellowfin_tuna",
    }

MODEL_NAME = 'efficientnet'

# @st.cache()
# def augment_model(efficientnet):
#     efficientnet.classifier[-1] = nn.Linear(in_features=1792, out_features=len(label_map), bias=True)
#     return efficientnet

with st.spinner('Model is being loaded..'):
    PATH = Path(__file__).resolve().parent.parent/'models'/'efficientnet_10_25_full.pt'
    # Use cuda to enable gpu usage for pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if MODEL_NAME in 'efficientnet':
    #     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)

    #     model_ft = augment_model(efficientnet)
    #     model_ft.load_state_dict(torch.load(PATH,map_location=device))

    model_ft = torch.load(PATH,map_location=device)

st.write("""
         # Endangered Fish Classification
         """
         )
st.write('Nearly half of the world depends on seafood for their main source of protein. In the Western and Central Pacific, where 60% of the world’s tuna is caught, illegal, unreported, and unregulated fishing practices are threatening marine ecosystems, global seafood supplies and local livelihoods. The Nature Conservancy is working with local, regional and global partners to preserve this fishery for the future.')
st.write('Currently, the Conservancy is looking to the future by using cameras to dramatically scale the monitoring of fishing activities to fill critical science and compliance monitoring data gaps. Our trained model helps to identify when target endangered species have been caught by fishermen.')
file = st.file_uploader("Please upload your fish image", type=["jpg","png"])


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache()
def import_and_predict(image_data: Image.Image, model, k: int, index_to_label_dict: dict)-> list:
    
    if MODEL_NAME in 'vgg':
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                )
    if MODEL_NAME in 'resnet' or MODEL_NAME in 'alexnet' or MODEL_NAME in 'efficientnet':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                    )
                        
    actual_img = transform(image_data).to(device)
    actual_img = actual_img.unsqueeze(0) # add one dimension to the front to account for batch_size

    formatted_predictions = model(actual_img)
    return formatted_predictions

if file is None:
    pass
else:
    image = Image.open(file)

    st.image(image, use_column_width=True)
    
    model_ft.eval()
    predictions = import_and_predict(image, model_ft, k = 3, index_to_label_dict = label_map) 

    predicted_fish = label_map[int(torch.argmax(predictions))]
    normalised_list = torch.nn.functional.softmax(predictions, dim = 1)
    values, indices = torch.topk(normalised_list, 3)

    st.title('The predicted fish is: ' + predicted_to_actual_dict[predicted_fish])

    st.title('Here are the three most likely fish species(click for more info!)')
    df = pd.DataFrame(data=np.zeros((3, 2)),
                      columns=['Species', 'Confidence Level'],
                      index=np.linspace(1, 3, 3, dtype=int))

    # print(values.detach().numpy()[0][1])

    for count, i in enumerate(values.detach().numpy()[0]):
        x = int(indices.detach().numpy()[0][count])
        df.iloc[count, 0] = f'<a href="{fish_to_wiki[x]}" target="_blank">{predicted_to_actual_dict[label_map[x]].title()}</a>'
        df.iloc[count, 1] = np.format_float_positional(i, precision=8)

    st.write(df.to_html(escape=False, justify = 'left'), unsafe_allow_html=True)
    if predicted_fish not in ['OTHER', 'Nof']:

        PATH_fish = Path(__file__).resolve().parent.parent/'data'/'fishes_ref'/ (predicted_fish + '.jpg')
        st.title('Here is a sample image of ' + predicted_to_actual_dict[predicted_fish])
        reference_image = Image.open(PATH_fish)
        st.image(reference_image)
    



    
    