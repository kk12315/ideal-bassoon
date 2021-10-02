import time
import numpy as np
import cv2
import json
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from char_dict import dict_map


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

wrong_answer = []
cnt = 0
for file_path in file_list:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = prerocess(img)
    
    prediction = model.predict(tf.expand_dims(img, 0))[0]
    scores = tf.nn.softmax(prediction)

    label = os.path.basename(os.path.dirname(file_path))
    gt = label
    name, prob = class_list[np.argmax(scores)], 100*np.max(scores)
    print("예측된 캐릭터 {} 실제 라벨 {}, 확률 {}, 예측된 한글{} 파일명 {}".format(name,gt,prob, dict_map[name], file_path))
    cnt +=1
    if name != gt:
        wrong_answer.append([name,gt,prob, dict_map[name], file_path])

print(" \n\n\n\n")

print("총 진행된 갯수 {} 오답수 {} 정확도 {}".format(cnt, len(wrong_answer), len(wrong_answer)/cnt *100))

print(" 오답리스트 ---------")



for data in wrong_answer:
    name,gt,prob, dict_map, file_path = data
    print(name,gt,prob, file_path)