
# coding: utf-8

# ## Loading data from folder

# In[14]:

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

# In[47]:

import matplotlib.pyplot as plt
import random
#get_ipython().magic('matplotlib inline')
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
#display_im(images,measurements)
print('Size of measurements is',measurements.shape)
print('One of the measurements is:',measurements[100])
print('Size of image:',images[2].shape)
print('Size of images numpy array:',images.shape)


# ## Splitting dataset into Training ,Validation and test set

# In[48]:

from sklearn.model_selection import train_test_split

X=images
y=measurements

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.20, random_state=0)

print('X_train size = ',X_train.shape,'y_train size = ',y_train.shape)
print('X_valid size = ',X_valid.shape,'y_valid size = ',y_valid.shape)
print('X_test size = ',X_test.shape,'y_test size = ',y_test.shape)


# ## Algorithm

# In[49]:

from keras.models import Sequential
from keras.layers import Flatten,Dense

model=Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2)
model.save('model.h5')


# In[ ]:



