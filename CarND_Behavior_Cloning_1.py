
# coding: utf-8

# ## Loading data from folder

# In[1]:

import csv
import cv2
import numpy as np
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
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)


# ## Displaying an image and checking integrity of the dataset

# In[2]:

import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')
def display_im(X,y):
    index=random.randint(0,len(X))
    image=X[index].squeeze()
    plt.figure(figsize=(4,4))
    plt.imshow(image)
    print('Output Label for the input is :', y[index])
    

# Reshaping the images python list into numpy array
images=np.reshape(images,(-1,160,320,3))
measurements=np.reshape(measurements,(-1))


# Sample outputs from the dataset to check integrity
display_im(images,measurements)
print('Size of measurements is',measurements.shape)
print('One of the measurements is:',measurements[100])
print('Size of image:',images[2].shape)
print('Size of images numpy array:',images.shape)


# ## Splitting dataset into Training ,Validation and test set

# In[3]:

from sklearn.model_selection import train_test_split

X=images
y=measurements

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.20, random_state=0)

print('X_train size = ',X_train.shape,'y_train size = ',y_train.shape)
print('X_valid size = ',X_valid.shape,'y_valid size = ',y_valid.shape)
print('X_test size = ',X_test.shape,'y_test size = ',y_test.shape)


# ## Algorithm

# In[19]:

from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
# There was a problem with original Keras progress bar due to which Notebook used to hang.
# Changing the progress bar with another version of it so as to get the code working in Notebook. Look at model.fit command on how its used
#from keras_tqdm import TQDMNotebookCallback 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Cropping2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
model=Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape=(160,320,3)))

""""model.add(Conv2D(32, 3, 3, input_shape=(160, 320, 3),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())


model.add(Dense(1))
"""

#model.add(Conv2D(32, 3, 3, input_shape=(70, 25, 3),activation='relu'))
model.add(Conv2D(32, 3, 3, input_shape=(160, 320, 3),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
num_examples = len(X_train)
batch_size=4
epochs=3
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0] // batch_size,nb_epoch=epochs,validation_data=(X_test, y_test))

#model.fit(X_train,y_train,validation_split=0.2,nb_epoch=3, verbose=0, callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True)])
model.save('model.h5')


# In[ ]:



