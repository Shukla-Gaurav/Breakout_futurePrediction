import random
import os
import _pickle as cPickle
from os import listdir
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import csv

def dump_object(dump_file,obj):
    writer = open(dump_file, 'wb')
    cPickle.dump(obj, writer)
    writer.close()

def get_object(dump_file):
    reader = open(dump_file, 'rb')
    obj = cPickle.load(reader)
    reader.close()
    return obj

def preprocess_img(root_folder,dump_file):
    listPCAimages = []
    rewards = []
    for episode in listdir(root_folder):
        images_per_episode = []
        for img_file in listdir(root_folder + '/' + episode):
            if img_file.endswith(".csv"):
                #read labels from csv file 
                data = pd.read_csv(root_folder + '/' + episode +'/'+ img_file,header = None)
                reward = np.array(data.values).astype(int)

                #append one more reward for first image
                reward = np.vstack((np.array([[0]]),reward)).T[0]
                rewards.append(reward)
            else:
                img = Image.open(root_folder + '/' + episode +'/'+ img_file)
                img_mat = np.array(img)
                images_per_episode.append(img_mat.flatten())
        images_per_episode = np.array(images_per_episode)
        pca = PCA(n_components=50)
        processed_imgs = pca.fit_transform(images_per_episode)
        listPCAimages.append(processed_imgs)
        print(episode)
        break

    dump_object(dump_file,[listPCAimages,rewards])

def get_training_data(dump_file):
    #read pickle file
    processed_data = get_object(dump_file)
    listPCAimages,rewards = processed_data[0],processed_data[1]

    feature_data = []
    label_data = []

    count = 0
    for images,reward in zip(listPCAimages,rewards):
        print(reward.shape,":",count)
        count += 1
        
        for i in range(len(images)):
            #break if exceeds
            if(i+9 >= len(images)):
                break
            img_set = images[i:i+7]
            img_sampling = img_set[np.random.choice(img_set.shape[0], 5, replace=False), :]
            vec_img = img_sampling.flatten()

            #create training data
            feature_data.append(vec_img)
            label_data.append(reward[i+9])
    return feature_data,label_data

def train_data_csv(root_folder,dump_file):
    preprocess_img(root_folder,dump_file)
    feature_data,label_data = get_training_data(dump_file)

    feature_data = np.array(feature_data)
    label_data = np.array(label_data)
    label_data = np.reshape(label_data, (label_data.shape[0],1)) 

    print(feature_data.shape)
    print(label_data.shape)

    #merge feature and label data
    train_data = np.hstack((feature_data,label_data))

    with open(root_folder + '/train_data.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(train_data)
        print('Done')
    writeFile.close()

train_data_csv('/home/gaurav/Desktop/IITD_1stsem/ML/train_dataset',"processed_data.pkl")