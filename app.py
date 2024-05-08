import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import urllib.request
import requests
import streamlit as st
st.set_page_config(layout="wide",page_icon="🖼️", page_title="Brian Tumor Segmentation")
from ultralytics import YOLO
# Load a model



# read the shareable link for the model file from a text file
with open('model_url.txt', 'r') as f:
    model_url = f.read().strip()

@st.cache_resource
def load_cached_model(model_url):
    '''Download and cash the model from the model_url'''

    # download the model file from Google Drive
    url = 'https://drive.google.com/uc?id=' + model_url.split('/')[-2]
    print(url)
    model_filename, headers = urllib.request.urlretrieve(url)
    print(model_filename)
    # load the model from the downloaded file
    loaded_model = YOLO(model_filename, task='segment')

    return loaded_model


# load the cached model
model = load_cached_model(model_url)
#model = YOLO('best.pt', task='segment')


def predict_with_bounding_box(image, model=model):

    # Load the YOLOv8 model
    model = model

    # Read the image
    img = image

    # Make the prediction
    results = model(img)

    # Filter and draw bounding boxes
    for det in results[0].boxes:
        print(det.xyxy.tolist())

        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = det.xyxy.tolist()[0]

        # Draw bounding box and label (class name)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2) 

    return img


def main():
    '''main function for Streamlit app'''
    html_temp="""
                <div style="background: linear-gradient(to right, #ff5f6d, #ffc371); margin-bottom:20px;">
                <h2 style="color:white;text-align:center; font-family:unset;">Brian Tumor Segmentation </h2>
                </div>
              """
    html_temp1="""
                <div style="background: linear-gradient(to left, #ff5f6d, #ffc371); margin-top:100%;" >
                <h4 style="color:white;text-align:center; font-family:unset;">
                <a style="color:black; text-decoration:none; font-size:30px" href="https://github.com/Yousef-Nasr/human-segmentation">🚀 Git repository</a>
                </h4>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)
    r_image, l_image = st.columns(2)
    option = st.sidebar.radio('Select input type:', ('Upload', 'URL'))
    if option == 'Upload':
        uploaded_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            # display uploaded image
            r_image.image(uploaded_file, caption="original image", width=500)
            pil_img = Image.open(uploaded_file)
            pil_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            newimg = pil_img.copy()
            # if newimg.size > (500, 500):
            #     newimg = newimg.resize((500, 500))
    elif option == 'URL':
        url = st.sidebar.text_input('Enter an image URL')
        if url:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                r_image.image(img, caption="original image", width=500)
                newimg = img.copy()
                if newimg.size > (500, 500):
                    newimg = newimg.resize((500, 500))
            except:
                st.warning('URL is invalid', icon="🚨")
    done_btn = st.button('Predict 🧙‍♂️', use_container_width=True)

    if done_btn:
        print(newimg)
        result = predict_with_bounding_box(newimg)
        print('done')
        l_image.image(result)
        st.success('Done !', icon="✅")
    
    st.sidebar.markdown(html_temp1,unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()

