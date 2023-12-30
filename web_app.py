import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

model = tf.keras.models.load_model("model.h5")

# Title
st.markdown("<h1 style='text-align: center; color: #add8e6;'>WaterHub</h1>", unsafe_allow_html=True)

# About WaterHub Section
st.markdown("<h3 style='color: #add8e6;'>What is WaterHub?</h3>", unsafe_allow_html=True)

# Description
st.markdown("<p style='color: #add8e6;'>One of the largest issues that people in marginalized communities and in underdeveloped countries is a lack of adequate water quality. Thus, in lieu of UN's SDG #6 it is important to create a tool that can classify if the water that you are consuming is clean or not. Thus, in WaterHub I created a solution that allows you to take an image of the water cup that you have, and it can output if the water that you are drinking is clearn or dirty.</p>", unsafe_allow_html=True)

# Water Quality Testing Section
st.markdown("<h3 style='color: #add8e6;'>Water Quality Checker</h3>", unsafe_allow_html=True)

# Camera input
img_file_buffer = st.camera_input("Take a picture of your water cup")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 150))  # Resize the image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match the model's input shape

    # Predict
    output = model.predict(img)
    if(output[0][0] == 0):
        st.write("The water is clean!")
    else:
        st.write("The water is dirty!")

# Water Consumption Section
st.markdown("<h3 style='color: #add8e6;'>Water Consumption</h3>", unsafe_allow_html=True)

# Water Consumption Description
st.markdown("<p style='color: #add8e6;'>WaterHub also allows you to track your water consumption. Simply input the amount of water that you drink and it will be added to your total water consumption.</p>", unsafe_allow_html=True)

water_consumption = 0

# Water Consumption Input
water_consumption += st.number_input("Enter the number of cups of water you have drank today:", min_value=0, value=0)

# Water Consumption Output
st.write("You have drank", water_consumption, "cups of water today, and should drink another", 15.5 - water_consumption, "cups of water to meet the recommended amount of water consumption!")