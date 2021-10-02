#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE=224 # 모든 이미지는 IMG_SIZExIMG_SIZE으로 크기가 조정됩니다


train_ds = tf.keras.preprocessing.image_dataset_from_directory('cropped/dataset', validation_split=0.2, subset='training',
                                                               seed=123,image_size=(IMG_SIZE, IMG_SIZE),batch_size=4)
val_ds = tf.keras.preprocessing.image_dataset_from_directory('cropped/dataset', validation_split=0.2, subset='validation',
                                                               seed=123,image_size=(IMG_SIZE, IMG_SIZE),batch_size=4)

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label
import numpy as np
from PIL import Image

for img, label in train_ds.take(1):
    pass

train = train_ds.map(preprocess)
val = val_ds.map(preprocess)
BATCH_SIZE = 32

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

SHUFFLE_BUFFER_SIZE = 1000



IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

for image_batch, label_batch in train.take(1):
    pass

# 사전 훈련된 모델 MobileNet V2에서 기본 모델을 생성합니다.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
feature_batch = base_model(image_batch)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(len(train_ds.class_names))
prediction_batch = prediction_layer(feature_batch_average)

train = configure_for_performance(train)
val = configure_for_performance(val)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 50
validation_steps= 20

loss0,accuracy0 = model.evaluate(val, steps = validation_steps)

print("before training ... loss : {}, acc : {}".format(loss0, accuracy0))

history = model.fit(train,
                    epochs=initial_epochs,
                    validation_data=val)


print("training done ! evaluate model again")

loss0,accuracy0 = model.evaluate(val, steps = validation_steps)

print("after training ... loss : {}, acc : {}".format(loss0, accuracy0))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('training_history.png')
plt.show()


print("save fig ... file name : training_history.png")


# In[108]:


from tqdm import tqdm

cnt =0 
wrong_cnt = 0
for batch, label in tqdm(val):
    
    predictions = model.predict(batch)
    scores = tf.nn.softmax(predictions, axis=1)
    label_list = []
    for idx, score in enumerate(scores):
        cnt+=1
        predicted_label = np.argmax(score)
        gt = np.array(label[idx]).item()
        
        name, prob = train_ds.class_names[np.argmax(score)], 100*np.max(score)
        label_list.append([name, prob, train_ds.class_names[np.array(label[idx]).item()]])
        if predicted_label !=gt:
            wrong_cnt+=1
            print("같지 않은 데이터")
            print(predicted_label, gt , prob)
        
print("총 갯수 {}, 틀린 갯수{}".format(cnt, wrong_cnt))            
  
        

model.save('trained.h5')


with open('class_names.txt', 'w') as f:
    for item in train_ds.class_names:
        f.writelines(item)
        f.write('\n')