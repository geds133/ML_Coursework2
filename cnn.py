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
a = os.chdir(r'/home/c1422205/Documents/Modules/ml')
birds = pd.read_csv('bird_labs.csv', low_memory=False)


## get current directory
a = os.getcwd()

## create lists
label_data = []
IDs = []

df = pd.DataFrame(
    {'class_id': [], 'image': [], 'rotated_image': [], 'horizontal_flip_image': [], 'vertical_flip_image': []})

for index, row in birds.iterrows():
    class_name = row['class_name']
    class_id = row['class_id']
    image_name = row['image_name']
    x = row['x']
    y = row['y']
    width = row['width']
    height = row['height']

    ## access directory for the bird image
    c = os.path.join(a, 'CUB_200_2011', 'CUB_200_2011', 'images', class_name)
    os.chdir(c)

    image = Image.open(image_name)

    ## cropping parameters
    left = x
    top = y
    right = (x + width)
    bottom = (y + height)

    ## final size of cropped images
    x_crop = 75 #Here to increase image dimensions
    y_crop = 75
    newsize = (x_crop, y_crop)

    ## crop and resize
    crop_image = image.crop((left, top, right, bottom))
    new_image = crop_image.resize(newsize)
    #     new_image = image.resize(newsize)

    x = np.random.randint(0, 360)

    vertical = new_image.transpose(Image.FLIP_LEFT_RIGHT)
    horizontal = new_image.transpose(Image.FLIP_TOP_BOTTOM)
    rotated = new_image.rotate(x, expand=False)

    b = {'class_id': np.array(class_id - 1), 'image': np.array(new_image), 'rotated_image': np.array(rotated),
         'horizontal_flip_image': np.array(horizontal), 'vertical_flip_image': np.array(vertical)}

    df = df.append(b, ignore_index=True)

## train test split (20% test)
# SAM IF YOU COULD INSERT THE LINE USED FR SPLTTING TO 16 CLASSES
train, test = train_test_split(df, test_size=0.2)

train_IDs = []
test_IDs = []
train_list = []
test_list = []
train_numpy_label_data = []
test_numpy_label_data = []

for row in train.itertuples():
    ID = row.class_id
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)

    image = row.image
    rotated_image = row.rotated_image
    horizontal_flip_image = row.horizontal_flip_image
    vertical_flip_image = row.vertical_flip_image

    train_list.append(image)
    train_list.append(rotated_image)
    train_list.append(horizontal_flip_image)
    train_list.append(vertical_flip_image)

for row in test.itertuples():
    ID = row.class_id
    test_IDs.append(ID)
    image = row.image
    test_list.append(image)

train_IDs = np.array(train_IDs)
train_label_one_hot = to_categorical(train_IDs)
test_IDs = np.array(test_IDs)
test_label_one_hot = to_categorical(test_IDs)

for i in range(len(train_label_one_hot)):
    train_numpy_label_data.append((train_list[i], train_label_one_hot[i]))
for i in range(len(test_label_one_hot)):
    test_numpy_label_data.append((test_list[i], test_label_one_hot[i]))

## shuffle data
shuffle(train_numpy_label_data)
shuffle(test_numpy_label_data)

## removes the grey scale images *as shape is (50,50) not (50,50,3)*
for i in train_numpy_label_data:
    if i[0].shape == (x_crop, y_crop):
        train_numpy_label_data = [i for i in train_numpy_label_data if i[0].shape != (x_crop, y_crop)]

for i in test_numpy_label_data:
    if i[0].shape == (x_crop, y_crop):
        test_numpy_label_data = [i for i in test_numpy_label_data if i[0].shape != (x_crop, y_crop)]

pre_X_train = []
pre_y_train = []
pre_X_test = []
pre_y_test = []

## create X and y sets
for i in train_numpy_label_data:
    pre_X_train.append(i[0])
    pre_y_train.append(i[1])
for i in test_numpy_label_data:
    pre_X_test.append(i[0])
    pre_y_test.append(i[1])

X_train = np.asarray(pre_X_train) / 255
X_test = np.asarray(pre_X_test) / 255
y_train = np.asarray(pre_y_train)
y_test = np.asarray(pre_y_test)

plt.imshow(X_train[42], cmap = 'gist_gray')

####  Investigation into the shapes and image dimension ####

## X_train sizes
X_train.shape


#### Convolutional Neural Network ####

#information
batch_size = 32
num_classes = 200
epochs = 12

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import categorical_accuracy
import matplotlib.pyplot as plt

## Basic CNN architecture


# cnn = Sequential()
# cnn.add(Conv2D(50, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
# cnn.add(MaxPooling2D(pool_size=(2, 2)))
# cnn.add(Conv2D(25, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
# cnn.add(MaxPooling2D(pool_size=(2, 2)))
# cnn.add(Dropout(0.2))
#
# cnn.add(Flatten())
#
# cnn.add(Dense(400, activation='relu'))
# cnn.add(Dense(200, activation='softmax'))
#
# cnn.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])

## More layers and more complex CNN ##
cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(50, 50, 3))),
cnn.add(MaxPooling2D(2)),

cnn.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")),
cnn.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")),
cnn.add(MaxPooling2D(2)),

cnn.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")),
cnn.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")),
cnn.add(MaxPooling2D(2)),
cnn.add(Dropout(0.2)),
cnn.add(Flatten()),

cnn.add(Dense(600, activation="relu")),
cnn.add(Dropout(0.5)),

cnn.add(Dense(400, activation="relu")),
cnn.add(Dropout(0.5)),

cnn.add(Dense(200, activation="softmax")),

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#Training the model
history = cnn.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.33)

## Lines below save and load models which will be far easier when training to load.
#cnn.save(r'/home/c1422205/Documents/Modules/ml/model.h5')
#cnn = tf.keras.models.load_model(r'/home/c1422205/Documents/Modules/ml/model.h5')
cnn.summary()

# make predictions and evaluate accuracy
accuracy = cnn.evaluate(x=X_test,y=y_test, batch_size=32)
print("Accuracy: ",accuracy[1])

predictions = cnn.predict_classes(X_test)
np.unique(predictions)


#Plotting the treain, validation accuracy at each epoch
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
