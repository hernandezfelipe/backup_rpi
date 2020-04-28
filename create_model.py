import os
import numpy as np
import keras
from keras.models import Model
from keras import regularizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, GlobalMaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import MobileNet, InceptionV3
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
import psutil
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications import MobileNet, ResNet50, VGG16, NASNetMobile
from keras.regularizers import l2
import sys
from keras.models import model_from_json
from random import randint
from PIL import Image
from random import sample
import pandas as pd
from keras import backend as K
from keras import regularizers


HEIGHT = 32
WIDTH = 32
bs = 32

p = psutil.Process(os.getpid())

p.nice(psutil.HIGH_PRIORITY_CLASS)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = Sequential()
model.add(Conv2D(256,(3,3),padding='same', input_shape=(HEIGHT,WIDTH,1), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(512,(3,3),padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

opt = RMSprop(lr = 1e-4)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
dc = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-12)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
path = os.getcwd() + "/dataset"

datagen=ImageDataGenerator(
    zoom_range=0.2,
    shear_range=20,
    rotation_range=20,
    brightness_range=(0.5,1.5),
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    rescale=1./255.)

test_datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_directory(
    directory=path+'/training_set/',
    batch_size=bs,
    color_mode='grayscale',
    shuffle=True,
    class_mode="binary",
    target_size=(HEIGHT,WIDTH))

valid_generator=test_datagen.flow_from_directory(
    directory=path+'/validation_set/',
    batch_size=bs,
    color_mode='grayscale',
    shuffle=True,
    class_mode="binary",
    target_size=(HEIGHT,WIDTH))

test_generator=test_datagen.flow_from_directory(
    directory=path+'/test_set/',
    batch_size=bs,
    color_mode='grayscale',
    shuffle=True,
    class_mode="binary",
    target_size=(HEIGHT,WIDTH))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

from collections import Counter
itemCt = Counter(train_generator.classes)
maxCt = float(max(itemCt.values()))
cw = {clsID : maxCt/numImg for clsID, numImg in itemCt.items()}
print(cw)

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        print("Lr :", K.eval(self.model.optimizer.lr))
  
cv = myCallback()

model.fit_generator(generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=20,
        verbose = 1,
        callbacks=[cv, dc],
        workers = 8,
        max_queue_size = 1000000,
        class_weight = cw
        )


model.save_weights("best_model.h5")
model.save('./model.h5')





