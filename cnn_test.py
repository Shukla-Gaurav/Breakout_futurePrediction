import random
import os
import _pickle as cPickle
from os import listdir
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import csv
import sys
import itertools
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import gzip
from keras.models import load_model

def preprocess_img(root_folder):
    csv_data = [['id','Prediction']]
    list_episodes = sorted(listdir(root_folder))
    epi_count = 0 
    
    #process episodes for training data
    for episode in list_episodes:
        epi_count += 1
        images_per_episode = []
        images_folder = sorted(listdir(root_folder + '/' + episode))

        for img_file in images_folder:
            if img_file.endswith(".png"):
                imagepil = Image.open(root_folder + '/' + episode +'/'+ img_file)
                imagepil = np.array(imagepil.convert('L')) /255
                images_per_episode.append(imagepil)
                
        images_per_episode = np.array(images_per_episode)
        image_stack = np.stack(images_per_episode, axis = 2)
        image_stack = image_stack.reshape(1,210,160,5)
        classes = model.predict(image_stack)
        pred_cls = np.argmax(classes)
        csv_data.append([epi_count-1,pred_cls])
        print("Currently working on episode:", epi_count)
    return csv_data

if __name__ == '__main__':
    
    test_folder = sys.argv[1]
    model_file = sys.argv[2]
    model = load_model(model_file)
    model.summary()
    
    csv_data = preprocess_img(test_folder)

    ## Final write into the csv
    with open('kaggle_test.csv', 'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
        csvFile.close()



    





