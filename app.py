import numpy as np
from io import BytesIO
from PIL import Image
# import cv2
import urllib.request
import requests
import streamlit as st
st.set_page_config(layout="wide",page_icon="🖼️", page_title="Brain Tumor Segmentation")
from ultralytics import YOLO
# Load a model



# read the shareable link for the model file from a text file
with open('model_url.txt', 'r') as f:
    model_url = f.read().strip()

# @st.cache_resource
# def load_cached_model(model_url):
#     '''Download and cash the model from the model_url'''

#     # download the model file from Google Drive
#     url = 'https://drive.google.com/uc?id=' + model_url.split('/')[-2]
#     print(url)
#     model_filename, headers = urllib.request.urlretrieve(url)
#     print(model_filename)
#     # load the model from the downloaded file
#     loaded_model = YOLO(model_filename, task='segment')

#     return loaded_model


# load the cached model
#model = load_cached_model(model_url)
model = YOLO('best.pt', task='segment')


def predict_with_bounding_box(image, model=model):

    # Load the YOLOv8 model
    model = model

    # Read the image
    img = image

    # Make the prediction
    results = model(img)

    for i, r in enumerate(results):
        # Get annotated image as NumPy array (BGR format)
        annotated_img_bgr = r.plot(conf=False, labels=False)

        # Convert to RGB format (optional, if needed)
        annotated_img_rgb = annotated_img_bgr[..., ::-1]


    return annotated_img_bgr

def main():
    '''main function for Streamlit app'''
    html_temp="""
                <div style="background: linear-gradient(to right, #ff5f6d, #ffc371); margin-bottom:20px;">
                <h2 style="color:white;text-align:center; font-family:unset;">Brain Tumor Segmentation </h2>
                </div>
              """
    html_temp1="""
                <div style="background: linear-gradient(to left, #ff5f6d, #ffc371); margin-top:50%;" >
                <h4 style="color:white;text-align:center; font-family:unset;">
                <a style="color:black; text-decoration:none; font-size:30px" href="https://github.com/Yousef-Nasr/Brain-tumor-segmentation">🚀 Git repository</a>
                </h4>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)
    r_image, l_image = st.columns(2)
    uploaded_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file == None:
        r_image.image('img1.jpg', caption="original image", width=500)
        pil_img = Image.open('img1.jpg')
        newimg = pil_img.copy()

    if uploaded_file:
        # display uploaded image
        r_image.image(uploaded_file, caption="original image", width=500)
        pil_img = Image.open(uploaded_file)
        newimg = pil_img.copy()

    # make prediction
    done_btn = st.button('Predict 🧙‍♂️', use_container_width=True)

    if done_btn:
        result = predict_with_bounding_box(newimg)
        l_image.image(result, caption="Predicted image", width=500)
        st.success('Done !', icon="✅")
    
    st.sidebar.markdown(html_temp1,unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()

