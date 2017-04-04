
# coding: utf-8

# ## Loading data from folder

# In[30]:

# To convert notebook to python script use
# jupyter nbconvert --to script CarND_Behavior_Cloning.ipynb


import csv
import cv2
import numpy as np
import random
from PIL import Image



def read_data_udacity():
    lines=[]
    with open('../CarND-Behavioral-Cloning-P3/Udacity_data/data/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images=[]
    measurements=[]
    for line in lines[1:]:
        source_path=line[0]
        filename=source_path.split('/')[-1]
        current_path='../CarND-Behavioral-Cloning-P3/Udacity_data/data/IMG/'+filename
        image=cv2.imread(current_path)
        image=image[65:150,0:320]
        image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
        images.append(image)
        measurement=float(line[3])
        measurements.append(measurement)
    return lines,images,measurements,image

def read_data_recorded(images,measurements):
    lines_1=[]
    with open('../CarND-Behavioral-Cloning-P3/recorded_data/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines_1.append(line)

    for line in lines_1[1:]:
        source_path=line[0]
        fileloc=(source_path.split("\\"))
        filename=fileloc[8]
        #print(filename)
        current_path='../CarND-Behavioral-Cloning-P3/recorded_data/IMG/'+filename
        image=cv2.imread(current_path)
        image=image[65:150,0:320]
        image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
        images.append(image)
        measurement=float(line[3])
        measurements.append(measurement)
    
    images.extend([np.fliplr(img) for img in images])
    measurements.extend([-angle for angle in measurements])
    return images,measurements

def read_images(sample_dt):
    images=[]
    measurements=[]
    for line in sample_dt:
        source_path=line[0]
        if 'C:\\' in source_path:
            #print(line)
            source_path=line[0]
            #print(source_path)
            filename=(source_path.split("\\"))[8]
            #print(filename)
            current_path='../CarND-Behavioral-Cloning-P3/recorded_data/IMG/'+filename
            image=cv2.imread(current_path)
            image=image[65:150,0:320]
            image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
            images.append(image)
            measurement=float(line[3])
            measurements.append(measurement)
        else:
            #print('other')
            filename=source_path.split('/')[-1]
            current_path='../CarND-Behavioral-Cloning-P3/Udacity_data/data/IMG/'+filename
            image=cv2.imread(current_path)
            image=image[65:150,0:320]
            image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
            images.append(image)
            measurement=float(line[3])
            measurements.append(measurement)
    #image1 = Image.open(current_path)
    #image_array=image1.crop((0,65,320,150))
    #image_array=image_array.resize((200,66))
    #plt.imshow(np.asarray(image_array))
    images.extend([np.fliplr(img) for img in images])
    measurements.extend([-angle for angle in measurements])
    return images,measurements

def display_im(X,y):
    index=random.randint(0,len(X))
    image=X[index].squeeze()
    plt.figure(figsize=(4,4))
    #plt.imshow(image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('Output Label for the input',index,' is :', y[index])
    
def color_change(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2BGR)
    return image1


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)
    return image

def random_append_augment_images(images,measurements):
    rand_arr=random.sample(list(range(1,len(measurements))),int(len(measurements)/5))
    rand_arr_1=random.sample(list(range(1,len(measurements))),int(len(measurements)/5))
    for index,value in enumerate(measurements):
        if index in rand_arr:
            images.append(augment_brightness_camera_images(images[index]))
            measurements.append(measurements[index])
        if index in rand_arr_1:
            images.append(add_random_shadow(images[index]))
            measurements.append(measurements[index])
    return images, measurements

def img_resize(images):
    for img in images:
        img=cv2.resize(img,(200,66),interpolation = cv2.INTER_AREA)
        image=img
    return images,image

"""
lines, images, measurements, image=read_data_udacity()
meas_len=len(measurements)     
images, measurements=read_data_recorded(images,measurements)
print(len(images))
images, measurements=random_append_augment_images(images,measurements)
#images,image=img_resize(images)
print(len(images))
"""
"""
# Adding additional dataset
for index, value in enumerate(measurements[1:int(meas_len/4)]):
    if value !=0:
        measurements.append(-value)
        flip_image=cv2.flip(images[index],1)
        images=np.append(images,flip_image[None,...],0)
        if index%500==0:
            print(index)
"""





# In[31]:

def read_csv():
    with open('../CarND-Behavioral-Cloning-P3/Udacity_data/data/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    with open('../CarND-Behavioral-Cloning-P3/recorded_data/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def data_gen(lines, batch_size=30):
    while True:
        sample_dt=random.sample(lines[2:],int(batch_size/2))
        images,measurements=read_images(sample_dt)
        #for img in images:
        #    img=color_change(img)
        images,measurements=random_append_augment_images(images,measurements)
        
        images=np.asarray(images)
        measurements=np.asarray(measurements)
        #print('images shape',images.shape)
        #print('measurements shape',measurements.shape)
        ind = np.random.choice(images.shape[0], batch_size, replace=False)
        images=images[ind,:,:,:]
        measurements=measurements[ind]
        X=images
        y=measurements
        
        yield X,y
        


# In[32]:

lines=[]
lines=read_csv()



gen_example=data_gen(lines,40)
images,measurements=next(gen_example)
#images,measurements=gen_example

#print(images.shape)
image=images[5]
#print(image[1:10,1:10,1:2])
#for img in images:
#    img=cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    
#model.fit_generator(gen_train,samples)


# ## Displaying an image and checking integrity of the dataset

# In[29]:

import matplotlib.pyplot as plt
import random
#%matplotlib inline

#print(image.shape)
#images=np.asarray(images)
#measurements=np.asarray(measurements)
# Reshaping the images python list into numpy array
images=np.reshape(images,(-1,image.shape[0],image.shape[1],3))
measurements=np.reshape(measurements,(-1))

# Sample outputs from the dataset to check integrity
#display_im(images,measurements)
print('Size of measurements is',measurements.shape)
print('One of the measurements is:',measurements[4])
print('Size of image:',images[2].shape)
print('Size of images numpy array:',images.shape)


# In[24]:




# In[11]:




# ## Splitting dataset into Training ,Validation and test set

# In[61]:




# In[ ]:




# ## Algorithm

# In[56]:

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
#num_examples = len(X_train)
batch_size=30
epochs=4
gen_train=data_gen(lines,30)
gen_valid=data_gen(lines,30)


#chkpnt = ModelCheckpoint('checkpoints/weights2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=False)
#datagen.fit(X_train)
model.fit_generator(gen_train, samples_per_epoch=25000,nb_epoch=epochs,validation_data=gen_valid,nb_val_samples=2500, max_q_size=25, nb_worker=4, pickle_safe=True)

#print(model.evaluate(X_test,y_test))
#model.fit(X_train,y_train,validation_split=0.2,nb_epoch=3, verbose=0, callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True)])
model.save('model.h5')


# In[ ]:



