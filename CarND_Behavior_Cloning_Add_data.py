
# coding: utf-8

# In[10]:

from support_fnc import data_gen,read_images,augment_brightness_camera_images,add_random_shadow,random_append_augment_images,read_csv,img_resize
import csv
import cv2
import numpy as np
import random
from PIL import Image
from keras.models import load_model


# In[11]:

lines=[]
def read_csv():
    with open('../CarND-Behavioral-Cloning-P3/recorded_data_add/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    #with open('../CarND-Behavioral-Cloning-P3/recorded_data_jungle/driving_log.csv') as csvfile:
     #   reader=csv.reader(csvfile)
     #   for line in reader:
     #       lines.append(line)
    return lines


# In[12]:

lines=read_csv()


gen_example=data_gen(lines,40)
images,measurements=next(gen_example)
#images,measurements=gen_example

#print(images.shape)
image=images[5]


# In[ ]:

#model = load_model('model.h5')

from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
# There was a problem with original Keras progress bar due to which Notebook used to hang.
# Changing the progress bar with another version of it so as to get the code working in Notebook. Look at model.fit command on how its used
#from keras_tqdm import TQDMNotebookCallback 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Cropping2D, ELU
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# NVIDIA Model
input_shape=(image.shape[0],image.shape[1],3)
model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.50,input_shape=input_shape))
model.add(Conv2D(3,1,1,subsample=(1,1),border_mode="valid",init='he_normal'))
model.add(ELU())
model.add(Conv2D(24,5,5,subsample=(2,2),border_mode="valid",init='he_normal'))
model.add(ELU())
model.add(Conv2D(36,5,5,subsample=(2,2),border_mode="valid",init='he_normal'))
model.add(ELU())
model.add(Dropout(.4))
model.add(Conv2D(48,5,5,subsample=(2,2),border_mode="valid",init='he_normal'))
model.add(ELU())
model.add(Conv2D(64,3,3,subsample=(1,1),border_mode="valid",init='he_normal'))
model.add(ELU())
model.add(Conv2D(64,3,3,subsample=(1,1),border_mode="valid",init='he_normal'))
model.add(Dropout(.3))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1164,init='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(100,init='he_normal'))
model.add(ELU())
model.add(Dense(50,init='he_normal'))
model.add(ELU())
model.add(Dense(10,init='he_normal'))
model.add(ELU())
model.add(Dense(1,init='he_normal'))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])



batch_size=30
epochs=6
gen_train=data_gen(lines,30)
gen_valid=data_gen(lines,30)


model.fit_generator(gen_train, samples_per_epoch=25000,nb_epoch=epochs,validation_data=gen_valid,nb_val_samples=2500, max_q_size=25, nb_worker=4, pickle_safe=True)
model.save('model_add.h5')

