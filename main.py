import streamlit as st
import tensorflow as tf
import numpy as np
import openai
import os
import json
from groq import Groq

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
st.set_page_config(
    page_title="Satan 3.1 Chat",
    page_icon="𖤐",
    layout="centered"
)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Crop Disease Recognition","Chatbot"])

if(app_mode=="Home"):
    st.header("CROP DOCTOR")
    image_path = "crops-growing-in-thailand.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Crop Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying crop
    diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 images)
    3. Test (33 images)
    """)                    

elif (app_mode=="Crop Disease Recognition"):
    st.header("Crop Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if (st.button("Show image")):
        st.image(test_image,use_column_width=True)
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
elif(app_mode=="Chatbot"):
    working_dir=os.path.dirname(os.path.abspath("D:\Copy_of_system\dataset_3\config.json"))
    config_data=json.load(open(f"{working_dir}/config.json"))
    GROQ_API_KEY=config_data["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"]=GROQ_API_KEY
    client=Groq()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.title("𖤐 SATAN CHATBOT")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt= st.chat_input("Ask Satan Chatbot about Crop Disease Solutions")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user","content": user_prompt})
    messages=[
        {"role": "system","content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]




    response=client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response= response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant","content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)




    