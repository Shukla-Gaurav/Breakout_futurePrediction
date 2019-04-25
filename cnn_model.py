import cv2
import os
import numpy as np
import pandas as pd
import random
import time
from sklearn.metrics import f1_score
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
import sys
import _pickle as cPickle
from keras.utils import Sequence

import keras
from keras.models import Sequential

from keras.optimizers import Adam
import warnings
import gzip
warnings.filterwarnings('ignore')
from keras.layers import Dropout,Flatten,Dense


from keras.layers import Input
from keras.models import load_model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import Sequence

pickle_folder = sys.argv[1]
model_fileName = sys.argv[2]
try:
    modelFlag = sys.argv[3]
except:
    modelFlag = False

pickle_files = sorted(os.listdir(pickle_folder))
#print(pickle_files)
x_train = []
y_train = []

def giveRandomEpisodeData(random_episode):
    global x_train
    global y_train
    #random_episode = random.choice(pickle_files)
    print(random_episode)
    load_file_path = os.path.join(pickle_folder, random_episode)
    with gzip.open(load_file_path, 'rb') as f:
        loaded_object = cPickle.load(f)
    feature_data = loaded_object[0]
    label_data = loaded_object[1]
    
    whole_data = []
    for i in range(len(feature_data)):
        if label_data[i] == 0:
            z = np.array([1,0])
        else:
            z = np.array([0,1])
        whole_data.append((feature_data[i],z))

    print("in")
    x_train =[]
    y_train = []
    for x,y in whole_data:
        x_train.append(x)
        y_train.append(y)
    print("out")
    try:
        return np.array(x_train, dtype=np.float)/255, np.array(y_train, dtype=np.int)
    except:
        print("rerturning None")
        print("\n ****************************** error in folder ******************* \n  :",  random_episode, "\n")
        return None,None



if modelFlag:
    print("Training the trained model ---- ")
    model = load_model(model_fileName)
else:    
    num_classes = 2

    # Model as mentioned 
    ## Have to see strides and padding
    model = Sequential()
    model.add(Conv2D(32,strides = 2, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(210,160,5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,strides = 2, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    # model = Sequential()
    # model.add(Conv2D(32, strides =2, kernel_size=(5,5),activation='relu',input_shape=(210,160,5)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D())

    # #convolution 2nd layer
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.1))


    # #Fully connected 1st layer
    # model.add(Flatten()) #7
    # model.add(Dense(2048,use_bias=False)) 
    # model.add(BatchNormalization())
    # model.add(Activation('relu')) 
    # model.add(Dropout(0.25))    

    # #Fully connected final layer
    # model.add(Dense(2)) 
    # model.add(Activation('softmax')) 

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    
for epoch in range(300):
    x, y = giveRandomEpisodeData(pickle_files[epoch])
    del x_train, y_train
    x_train = []
    y_train = []
    if x is not None:
        #print(np.array(x_train).shape, np.array(y_train).shape)
        model.fit(x, y, batch_size=128,epochs=epoch+1, initial_epoch=epoch,  shuffle=True)
        model.save(model_fileName)

    del x,y
    
