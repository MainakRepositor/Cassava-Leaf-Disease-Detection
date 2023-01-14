from models import resnext50_32x4d
from dataset import CassavaDataset, get_transforms, classes
from inference import load_state, inference
from utils import CFG
from grad_cam import SaveFeatures, getCAM, plotGradCAM
import gc
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from html_mardown import (
    app_off,
    app_off2,
    model_predicting,
    loading_bar,
    result_pred,
    image_uploaded_success,
    more_options,
    class0,
    class1,
    class2,
    class3,
    class4,
    s_load_bar,
    class0_side,
    class1_side,
    class2_side,
    class3_side,
    class4_side,
    unknown,
    unknown_side,
    unknown_w,
    unknown_msg,
)

# Enable garbage collection
gc.enable()

# Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

# Set App title
st.title("Cassava disease prediction Web App")


# Set the directory path
my_path = "."

test = pd.read_csv(my_path + "/data/sample.csv")
img_1_path = my_path + "/images/img_1.jpg"
img_2_path = my_path + "/images/img_2.jpg"
img_3_path = my_path + "/images/img_3.jpg"
banner_path = my_path + "/images/banner.png"
output_image = my_path + "/images/gradcam2.png"


# Read and display the banner
st.sidebar.image(banner_path, use_column_width=True)

# App description
st.write(
    "The app predicts diseases in Cassava plants. The model was trained with the [cassava leaf disease dataset on Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/data). "
)
st.write(
    "**For more info:** [Blog Post](https://aminey.medium.com/how-to-train-ml-models-with-mislabeled-data-cf4bb353b3d9?sk=9f4ce905cd5c4f2d86ec3bf7b93d024c) **|** **Code:** [Github repository](https://github.com/Amiiney/cld-app-streamlit) **|**"
)
st.markdown("***")


# Set the selectbox for demo images
st.write("**Select an image for a DEMO**")
menu = ["Select an Image", "Image 1", "Image 2", "Image 3"]
choice = st.selectbox("Select an image", menu)


# Set the box for the user to upload an image
st.write("**Upload your Image**")
uploaded_image = st.file_uploader(
    "Upload your image in JPG or PNG format", type=["jpg", "png"]
)


# DataLoader for pytorch dataset
def Loader(img_path=None, uploaded_image=None, upload_state=False, demo_state=True):
    test_dataset = CassavaDataset(
        test,
        img_path,
        uploaded_image=uploaded_image,
        transform=get_transforms(data="valid"),
        uploaded_state=upload_state,
        demo_state=demo_state,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    return test_loader


# Function to deploy the model and print the report
def deploy(file_path=None, uploaded_image=uploaded_image, uploaded=False, demo=True):
    # Load the model and the weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(CFG.model_name, pretrained=False)
    states = [load_state(my_path + "/weights/resnext50_32x4d_fold0_best.pth")]

    # For Grad-cam features
    final_conv = model.model.layer4[2]._modules.get("conv3")
    fc_params = list(model.model._modules.get("fc").parameters())

    # Display the uploaded/selected image
    st.markdown("***")
    st.markdown(model_predicting, unsafe_allow_html=True)
    if demo:
        test_loader = Loader(img_path=file_path)
        image_1 = cv2.imread(file_path)
    if uploaded:
        test_loader = Loader(
            uploaded_image=uploaded_image, upload_state=True, demo_state=False
        )
        image_1 = file_path
    st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
    st.sidebar.image(image_1, width=301, channels="BGR")

    for img in test_loader:
        activated_features = SaveFeatures(final_conv)
        # Save weight from fc
        weight = np.squeeze(fc_params[0].cpu().data.numpy())

        # Inference
        logits, output = inference(model, states, img, device)
        pred_idx = output.to("cpu").numpy().argmax(1)

        # Grad-cam heatmap display
        heatmap = getCAM(activated_features.features, weight, pred_idx)

        ##Reverse the pytorch normalization
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        image = img[0] * STD[:, None, None] + MEAN[:, None, None]

        # Display image + heatmap
        plt.imshow(image.permute(1, 2, 0))
        plt.imshow(
            cv2.resize(
                (heatmap * 255).astype("uint8"),
                (328, 328),
                interpolation=cv2.INTER_LINEAR,
            ),
            alpha=0.4,
            cmap="jet",
        )
        plt.savefig(output_image)

        # Display Unknown class if the highest probability is lower than 0.5
        if np.amax(logits) < 0.57:
            st.markdown(unknown, unsafe_allow_html=True)
            st.sidebar.markdown(unknown_side, unsafe_allow_html=True)
            st.sidebar.markdown(unknown_w, unsafe_allow_html=True)

        # Display the class predicted if the highest probability is higher than 0.5
        else:
            if pred_idx[0] == 0:
                st.markdown(class0, unsafe_allow_html=True)
                st.sidebar.markdown(class0_side, unsafe_allow_html=True)
                st.write(" The predicted class is: **Cassava Bacterial Blight (CBB)**")
            elif pred_idx[0] == 1:
                st.markdown(class1, unsafe_allow_html=True)
                st.sidebar.markdown(class1_side, unsafe_allow_html=True)
                st.write(
                    "The predicted class is: **Cassava Brown Streak Disease (CBSD)**"
                )
            elif pred_idx[0] == 2:
                st.markdown(class2, unsafe_allow_html=True)
                st.sidebar.markdown(class2_side, unsafe_allow_html=True)
                st.write("The predicted class is: **Cassava Green Mottle (CGM)**")
            elif pred_idx[0] == 3:
                st.markdown(class3, unsafe_allow_html=True)
                st.sidebar.markdown(class3_side, unsafe_allow_html=True)
                st.write("The predicted class is: **Cassava Mosaic Disease (CMD)**")
            elif pred_idx[0] == 4:
                st.markdown(class4, unsafe_allow_html=True)
                st.sidebar.markdown(class4_side, unsafe_allow_html=True)
                st.write("The predicted class is: **Healthy**")

        st.sidebar.markdown(
            "**Scroll down to read the full report (Grad-cam and class probabilities)**"
        )

        # Display the Grad-Cam image
        st.title("**Grad-cam visualization**")
        st.write(
            "Grad-cam *(Class Acvitation Map)* highlights the important regions in the image for predicting the class concept. It helps to understand if the model based its predictions on the correct regions of the image."
        )
        gram_im = cv2.imread(output_image)
        st.image(gram_im, width=528, channels="RGB")

        # Display the class probabilities table
        st.title("**Class predictions:**")
        if np.amax(logits) < 0.57:
            st.markdown(unknown_msg, unsafe_allow_html=True)
        classes["class probability %"] = logits.reshape(-1).tolist()
        classes["class probability %"] = classes["class probability %"] * 100
        classes_proba = classes.style.background_gradient(cmap="Reds")
        st.write(classes_proba)
        del (
            model,
            states,
            fc_params,
            final_conv,
            test_loader,
            image_1,
            activated_features,
            weight,
            heatmap,
            gram_im,
            logits,
            output,
            pred_idx,
            classes_proba,
        )
        gc.collect()


# Set red flag if no image is selected/uploaded
if uploaded_image is None and choice == "Select an Image":
    st.sidebar.markdown(app_off, unsafe_allow_html=True)
    st.sidebar.markdown(app_off2, unsafe_allow_html=True)


# Deploy the model if the user uploads an image
if uploaded_image is not None:
    # Close the demo
    choice = "Select an Image"
    # Deploy the model with the uploaded image
    deploy(uploaded_image, uploaded=True, demo=False)
    del uploaded_image


# Deploy the model if the user selects Image 1
if choice == "Image 1":
    deploy(img_1_path)


# Deploy the model if the user selects Image 2
if choice == "Image 2":
    deploy(img_2_path)


# Deploy the model if the user selects Image 3
if choice == "Image 3":
    deploy(img_3_path)
