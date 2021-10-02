import time
import numpy as np
import cv2
import json
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)


IMG_SIZE = 224

def prerocess(image):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

with open('class_names.txt', 'r') as f:
    text = f.readlines()

def remove_str(text):
    text = text.replace("\n", "")
    return text
class_list = list(map(remove_str, text))

model = tf.keras.models.load_model('trained.h5')


# test

from glob import glob

file_list = glob('cropped/*/*/*.jpeg')

for file_path in file_list:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = prerocess(img)
    
    prediction = model.predict(tf.expand_dims(img, 0))[0]
    scores = tf.nn.softmax(prediction)

    label = os.path.basename(os.path.dirname(file_path))
    gt = label
    name, prob = class_list[np.argmax(scores)], 100*np.max(scores)
    print(name,prob,gt)
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         r = request
#         # convert string of image data to uint8
#         nparr = np.fromstring(r.data, np.uint8)
#         # decode image
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         return jsonify({'meta':detected['meta']})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8891)