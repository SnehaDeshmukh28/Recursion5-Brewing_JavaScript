import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

def image_processing(image):
    img_shape = (321, 321)
    classifier = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")(x))
    ])

    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    result = classifier.predict(img)
    return labels[np.argmax(result)], img1

def get_map(location):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(location)
    return location.address, location.latitude, location.longitude

def sidebar():
    st.sidebar.title("Travel Planner")
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About"])
    return page

def home():
    st.title("Landmark Recognition and Travel Guide")
    img = PIL.Image.open('logo.jpg')
    img = img.resize((256, 256))
    st.image(img)

    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        prediction, image = image_processing(save_image_path)
        st.image(image)
        st.header("📍 *Predicted Landmark is: " + prediction + '*')

        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: ' + address)
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader('✅ *Latitude & Longitude of ' + prediction + '*')
            st.json(loc_dict)

            m = folium.Map(location=[latitude, longitude], zoom_start=15)
            folium.Marker([latitude, longitude], popup=prediction).add_to(m)
            folium.CircleMarker(
                location=[latitude, longitude],
                radius=50,
                popup=prediction,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)

            st.subheader('✅ *' + prediction + ' on the Map*' + '🗺')
            folium_static(m)
        except Exception as e:
            st.warning("No address found!!")

def about():
    st.title("About Travel Planner")
    st.write("This is a simple Travel Planner application built with Streamlit.")
    st.write("Upload an image, and the app will recognize the landmark and provide information about it.")

def main():
    page = sidebar()
    if page == "Home":
        home()
    elif page == "About":
        about()

if __name__ == "__main__":
    main()
