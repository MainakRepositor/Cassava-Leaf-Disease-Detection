# cld-app-streamlit
 
 This project is a computer vision app, an end-to-end image classification web app with pytorch. The app classifies and predicts diseases in cassava plant images. The user has 2 options: Upload his own image or select an image for a demo. 
 
 Link to the Web App: https://share.streamlit.io/amiiney/cld-app-streamlit/main/app.py
 
 
 ![alt text](https://github.com/Amiiney/cld-app-streamlit/blob/main/images/cld-app.gif)
 
 The model was trained with the [Cassava leaf disease prediction dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification).
 
 ## How to run the app?
 All the necessary libraries are availbale in the [requirements.txt](https://github.com/Amiiney/cld-app-streamlit/blob/main/requirements.txt) file. You need to clone the repository, install requirements.txt and run streamlit.
 
 ```
 git clone https://github.com/Amiiney/cld-app-streamlit.git
 cd cld-app-streamlit
 pip install -r requirements.txt
 streamlit run app.py
 ```
 
## Reproducibility:
If you are looking for your first computer vision project, this app is a good starting point. The CLD web app can be reused in other image classification projects. You only need to change the model/weights with some modifications and benefit from the app's built-in features.

## Models:
The [models script](https://github.com/Amiiney/cld-app-streamlit/blob/main/models.py) has 3 models: Seresnext50_32_4d, ViT_base_16 an efficientnetB3 models. Only weights for seresnext50_32_4d are available for now. I will upload the other models weights in future commits.

## Features:
>The main User Interface features are:
* **The select image box:** For general audience to play with the app and discover all its features.
* **Upload image box:** For the targeted audience that would benefit from your app by uploading their own images.

>Prediction report features:
* **Predicting the correct class (disease):** The main functionality of the app!
* **Grad-Cam:** Outputs an image that highlights the main pixels that contain the class our model predicted. *(BUG TO BE FIXED)*
![alt text](https://github.com/Amiiney/cld-app-streamlit/blob/main/images/Screen%20Shot%202021-03-05%20at%2020.32.52.png)
* **Class probabilities:** Outputs a table with the class probability of each class, highliting the class with the highest probability.
![alt text](https://github.com/Amiiney/cld-app-streamlit/blob/main/images/Screen%20Shot%202021-03-05%20at%2020.33.05.png)

### Acknowledgement:
The dataset used to train the model is publicly available on Kaggle: [Cassava leaf disease prediction dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification).

The app is deployed with [streamlit sharing](https://streamlit.io/sharing). Streamlit is by far the best place to deploy and share machine learning web apps.

The inference code is generated and modified from many notebooks in the cassava leaf disease competition, mainly from the [Resnext50_32_4d inference notebook](https://www.kaggle.com/piantic/no-tta-cassava-resnext50-32x4d-inference-lb0-903) by [heroseo](https://www.kaggle.com/piantic).

The grad-cam code is generated and modified from the notebook [PANDA / PyTorch Grad-CAM](https://www.kaggle.com/yasufuminakama/panda-pytorch-grad-cam) by [Y. Nakama](https://www.kaggle.com/yasufuminakama).
