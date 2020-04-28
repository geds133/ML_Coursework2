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

birds = pd.read_csv('bird_labs.csv', low_memory=False)


## get current directory
a = os.getcwd()

## create lists
label_data = []
IDs = []

#### don't change this to change quantity of images, change at next part where said ###
df = pd.DataFrame(
    {'class_id': [], 'image': [], 'i1': [], 'i2': [], 'i3': [],'i4': [], 'i5': [], 'i6': [], 'i7': [], 'i8': [], 'i9': [], 'i10': [], 'i11': []})

for row in birds.itertuples():
    class_name = row.class_name
    class_id = row.class_id
    image_name = row.image_name
    x = row.x
    y = row.y
    width = row.width
    height = row.height

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
    
    ### change crop size from 40x40 - 90x90 by increments of 10
    x_crop = 70
    y_crop = 70
    newsize = (x_crop, y_crop)

    ## crop and resize
    crop_image = image.crop((left, top, right, bottom))
    new_image = crop_image.resize(newsize)
    #     new_image = image.resize(newsize)

    ### dont change this ###
    i1 = crop_image.rotate(np.random.randint(0, 45), expand=True).resize(newsize)
    i2 = crop_image.rotate(10, expand=True).resize(newsize)
    i3 = crop_image.rotate(-10, expand=True).resize(newsize)
    horizontal = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
    i4 = horizontal.resize(newsize)
    i5 = horizontal.rotate(10, expand=True).resize(newsize)
    i6 = horizontal.rotate(-10, expand=True).resize(newsize)
    i7 = horizontal.rotate(np.random.randint(0, 45),expand = True).resize(newsize)
    i8 = horizontal.rotate(np.random.randint(0, 45),expand = True).resize(newsize)
    i9 = horizontal.rotate(np.random.randint(0, 45),expand = True).resize(newsize)
    i10 = crop_image.rotate(np.random.randint(0, 45), expand=True).resize(newsize)
    i11 = crop_image.rotate(np.random.randint(0, 45), expand=True).resize(newsize)
    
    ### or this ###
    b = {'class_id': np.array(class_id -1), 'image': np.array(new_image), 'i1': np.array(i1),
         'i2': np.array(i2), 'i3': np.array(i3),'i4': np.array(i4),
         'i5': np.array(i5), 'i6': np.array(i6), 'i7': np.array(i7), 'i8':np.array(i8),
         'i9': np.array(i9), 'i10':np.array(i10), 'i11': np.array(i11)}
    


    df = df.append(b, ignore_index=True)

### these are to make data frames with smaller no. of classes ###
df10 = df.head(542)
# df20 = df.head(1114)
# df30 = df.head(1700)
# df40 = df.head(2289)   
# df50 = df.head(2889)

### change data frame input to according to the data frame above ###
## train test split (20% test)
train, test = train_test_split(df10, test_size=0.2)

train_IDs = []
test_IDs = []
train_list = []
test_list = []
train_numpy_label_data = []
test_numpy_label_data = []


### change the amount of images and IDs appended, to the amount of images you want ###
for row in train.itertuples():
    ID = row.class_id
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)
    train_IDs.append(ID)

    image = row.image
    a1 = row.i1
    a2= row.i2
    a3 = row.i3
    a4 = row.i4
    a5 = row.i5
    a6 = row.i6
    a7 = row.i7
    a8 = row.i8
    a9 = row.i9
    a10 = row.i10
    a11 = row.i11
    
    train_list.append(image)
    train_list.append(a1)
    train_list.append(a2)
    train_list.append(a3)
    train_list.append(a4)
    train_list.append(a5)
    train_list.append(a6)
    train_list.append(a7)
    train_list.append(a8)
    train_list.append(a9)
    train_list.append(a10)
    train_list.append(a11)
    

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
    

X_train = np.asarray(pre_X_train)/255
X_test = np.asarray(pre_X_test)/255
y_train = np.asarray(pre_y_train)
y_test= np.asarray(pre_y_test)
## X_train sizes
X_train.shape



#### Convolutional Neural Network ####


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalisation import BatchNormalisation
import matplotlib.pyplot as plt


cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(x_crop, y_crop, 3)))
cnn.add(BatchNormalisation())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
cnn.add(BatchNormalisation())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn.add(BatchNormalisation())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.2))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = cnn.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

#cnn.save(r'/home/c1422205/Documents/Modules/ml/model.h5')


#cnn = tf.keras.models.load_model(r'/home/c1422205/Documents/Modules/ml/model.h5')
cnn.summary()

# make predictions
accuracy = cnn.evaluate(x=X_test,y=y_test, batch_size=32)
print("Accuracy: ",accuracy[1])

predictions = cnn.predict_classes(X_test)
np.unique(predictions)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
