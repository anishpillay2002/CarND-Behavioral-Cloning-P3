
# coding: utf-8

# In[ ]:

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
        if 'recorded data\IMG' in source_path:
            #print('line',line)
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
        elif 'recorded_data_add\IMG' in source_path:
            #print(source_path)
            filename=(source_path.split("\\"))[8]
            #print(filename)
            current_path='../CarND-Behavioral-Cloning-P3/recorded_data_add/IMG/'+filename
            image=cv2.imread(current_path)
            image=image[65:150,0:320]
            image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
            images.append(image)
            measurement=float(line[3])
            measurements.append(measurement)
        elif 'recorded_data_jungle\IMG' in source_path:
            #print(source_path)
            filename=(source_path.split("\\"))[8]
            #print(filename)
            current_path='../CarND-Behavioral-Cloning-P3/recorded_data_jungle/IMG/'+filename
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
        



