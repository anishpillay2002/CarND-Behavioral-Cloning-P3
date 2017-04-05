
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


# In[ ]:

model = load_model('model.h5')

batch_size=30
epochs=4
gen_train=data_gen(lines,30)
gen_valid=data_gen(lines,30)


model.fit_generator(gen_train, samples_per_epoch=400,nb_epoch=epochs,validation_data=gen_valid,nb_val_samples=2500, max_q_size=25, nb_worker=4, pickle_safe=True)
model.save('model_add.h5')

