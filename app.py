import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask import Flask, render_template, request, redirect, url_for
import os
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import folium
import webbrowser

app = Flask(__name__)

# Load model and labels
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            save_image_path = './Uploaded_Images/' + img_file.filename
            img_file.save(save_image_path)
            prediction, image = image_processing(save_image_path)

            try:
                address, latitude, longitude = get_map(prediction)

                return render_template('result.html', 
                                       prediction=prediction, 
                                       image_path=url_for('static', filename=img_file.filename), 
                                       address=address, 
                                       latitude=latitude, 
                                       longitude=longitude)
            except Exception as e:
                return render_template('error.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Here you can handle the file upload process
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
