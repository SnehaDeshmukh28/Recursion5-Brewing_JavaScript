# import streamlit as st
# import PIL
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import pandas as pd
# from geopy.geocoders import Nominatim
# import folium
# from streamlit_folium import folium_static
# import google.generativeai as genai
# import subprocess

# # Landmark classification model details (replace with your actual model URL and label map)
# model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# labels_path = 'landmarks_classifier_asia_V1_label_map.csv'  # Adjust if your label map has a different name

# # Set up the Gemini API key and model configuration
# genai.configure(api_key="AIzaSyDbUQj2jSe1THDWuFVdGKRCJ7ozrzd1MyA")
# generation_config = {
#     "temperature": 0.9,
#     "top_p": 1,
#     "top_k": 1,
#     "max_output_tokens": 2048,
# }

# safety_settings = [
#     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
# ]

# model = genai.GenerativeModel(
#     model_name="gemini-1.0-pro",
#     generation_config=generation_config,
#     safety_settings=safety_settings,
# )

# def image_processing(image):
#     """Processes and classifies an uploaded image."""
#     img_shape = (321, 321)
#     classifier = tf.keras.Sequential([
#         tf.keras.layers.Lambda(lambda x: hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")(x))
#     ])

#     img = PIL.Image.open(image)
#     img = img.resize(img_shape)

#     # Ensure image has 3 channels (RGB) for the model
#     img = img.convert('RGB')

#     img1 = img
#     img = np.array(img) / 255.0
#     img = img[np.newaxis]
#     result = classifier.predict(img)

#     # Load label map (replace with your actual logic)
#     df = pd.read_csv(labels_path)
#     labels = dict(zip(df.id, df.name))

#     return labels[np.argmax(result)], img1

# def get_map(location):
#     """Retrieves location details using geopy."""
#     geolocator = Nominatim(user_agent="Your_Name")
#     location = geolocator.geocode(location)
#     return location.address, location.latitude, location.longitude

# def sidebar():
#     """Displays navigation options in the sidebar."""
#     st.sidebar.title("Travel Planner")
#     st.sidebar.subheader("Navigation")
#     page = st.sidebar.radio("Go to", ["Home", "About"])
#     return page

# def home():
#     """Main page for landmark recognition, map display, and itinerary generation."""
#     st.title("Landmark Recognition and Travel Guide")
#     img = PIL.Image.open('logo.jpg')  # Replace with your logo image path
#     img = img.resize((256, 256))
#     st.image(img)

#     img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
#     if img_file is not None:
#         save_image_path = './Uploaded_Images/' + img_file.name
#         with open(save_image_path, "wb") as f:
#             f.write(img_file.getbuffer())

#         prediction, image = image_processing(save_image_path)
#         st.image(image)
#         st.header(f" *Predicted Landmark is: {prediction}*")
#         prompt_text = f"Make a travel plan for {prediction}."

#         if st.button("Make a Plan"):
#             try:
#                 convo = model.start_chat(history=[])
#                 convo.send_message(prompt_text)
#                 st.success("Generated Text:")
#                 st.write(convo.last.text)
#             except Exception as e:
#                 st.warning("Error occurred while fetching locationdetails.")

#         # Display address and map
#         try:
#             address, latitude, longitude = get_map(prediction)
#             st.success('Address: ' + address)
#             loc_dict = {'Latitude': latitude, 'Longitude': longitude}
#             st.json(loc_dict)
#             st.markdown("<h1 style='text-align: center; color: black;'>make a plan!</h1>", unsafe_allow_html=True)

#             m = folium.Map(location=[latitude, longitude], zoom_start=15)
#             folium.Marker([latitude, longitude], popup=prediction).add_to(m)
#             folium.CircleMarker(
#                 location=[latitude, longitude],
#                 radius=50,
#                 popup=prediction,
#                 color='blue',
#                 fill=True,
#                 fill_color='blue'
#             ).add_to(m)

#             st.subheader('✅ *' + prediction + ' on the Map*' + '')
#             folium_static(m)
#         except Exception as e:
#             st.warning("No address found!!")

# def about():
#     st.title("About Travel Planner")
#     st.write("This is a simple Travel Planner application built with Streamlit.")
#     st.write("Upload an image, and the app will recognize the landmark and provide information about it.")

# def main():
#     page = sidebar()
#     if page == "Home":
#         home()
#     elif page == "Plan":
#         plan_page()
#     elif page == "About":
#         about()

# if __name__ == "__main__":
#     main()
