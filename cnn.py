##### NOTES
## this needs to be in a directory which also contains the CUB_200_2011 file and the bird_labs.csv to run
## crops the images to the bounding boxes provided, no parts are pinpointed on to the birds.
## splits the images into the numpy array and its label
## creates train and test (validation can be added in keras model)
## may have trouble with the tensorflow package import, ensure it is downloaded correctly

## crop size is changable by x_crop and y_crop variables (set to 50x50 pixels)
## can rotate/flip the images to provide extra images if more are required to get better accuracy
## 8 of the 11788 images are greyscale, so have I removed these as they cause problems in code and are insignificant
## y_train has length 201, as the id's start at 1 but the encoding method includes 0, so can ignore, or possibly remove if needed

## should take around 5-10 minutes to run

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import os
from PIL import Image
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.utils import to_categorical

## load data
os.chdir(r'/home/c1422205/Documents/Modules/ml')
birds = pd.read_csv('bird_labs.csv', low_memory=False)

## get current directory
a = os.getcwd()

## create lists
label_data = []
IDs = []
image_array_list = []
numpy_label_data = []

## iterate through the rows of the csv, to obtained cropped and resized image, and labels 
for index, row in birds.iterrows():
    
    class_name = row['class_name']
    class_id = row['class_id']
    image_name = row['image_name']
    x = row['x']
    y = row['y']
    width = row['width']
    height = row['height']
    
    ## access directory for the bird image 
    c = os.path.join(a,'CUB_200_2011','CUB_200_2011','images', class_name)
    os.chdir(c)
    
    image = Image.open(image_name)
    
    ## cropping parameters
    left = x
    top = y
    right = (x + width)
    bottom = (y + height)
    
    ## final size of cropped images 
    x_crop = 50
    y_crop = 50 
    newsize = (x_crop, y_crop) 

    ## crop and resize
    crop_image = image.crop((left, top, right, bottom))
    new_image = crop_image.resize(newsize)
#     new_image = image.resize(newsize)
    
    ## add numpy array of the class id to the IDs list for it to be binary-coded
    class_id = np.array(class_id)
    IDs.append(class_id)
    
    ## add image numpy array to a list with all image-arrays
    image_array_list.append(np.array(new_image))


## convert IDs list to numpy array and the convert labels to binary arrays (one-hot coding) 
IDs = np.array(IDs)
label_one_hot = to_categorical(IDs)

## rejoin image-array and binary label as tuple into the numpy_label_data list
for i in range(len(label_one_hot)):
    numpy_label_data.append((image_array_list[i],label_one_hot[i]))
    
## shuffle data 
shuffle(numpy_label_data)  

## removes the grey scale images *as shape is (50,50) not (50,50,3)*
for i in numpy_label_data:
    if i[0].shape != (x_crop,y_crop,3):
        numpy_label_data.remove(i)

X = []
y = []

## create X and y sets
for i in numpy_label_data:
    X.append(i[0])
    y.append(i[1])

## Create train and test set, then create validation set from the train set (gives 7544 train, 1886 val, 2358 test)
pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) 
#  this line is #'d out, as you can include validation set in the keras models when fittting, can include here if you want

## normalizes the RGB values to 0-1 instead of 0-255, and converts lists into numpy array 
X_train = np.asarray(pre_X_train)/255
X_test = np.asarray(pre_X_test)/255

y_train = np.asarray(pre_y_train)
y_test = np.asarray(pre_y_test)

## print to show example for format of the output of each variable
print('X_train example:','\n',X_train[0],'\n\n','y_train example:','\n',y_train[0])
plt.imshow(X_train[0])
plt.show()

plt.imshow(X_train[42], cmap = 'gist_gray')

####  Investigation into the shapes and image dimension  - need to run Practice script first  ####

## X_train sizes
X_train.shape
x_train.shape





#### Convolutional Neural Network ####

batch_size = 128
num_classes = 200
epochs = 12


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import categorical_accuracy


cnn = Sequential()
cnn.add(Conv2D(50, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.2))

cnn.add(Flatten())

cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(201, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = cnn.fit(X_train, y_train, epochs=35)
cnn.save(r'/home/c1422205/Documents/Modules/ml/model.h5')

accuracy = cnn.evaluate(x=X_test,y=y_test, batch_size=32)
print("Accuracy: ",accuracy[1])


X_train.shape
len(y_train.unique())
predictions = cnn.predict(X_test)
categorical_accuracy(y_test, predictions)