import random
import os
import _pickle as cPickle
from os import listdir
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def preprocess_img(root_folder):
    listPCAimages = []
    rewards = []
    for episode in listdir(root_folder):
        images_per_episode = []
        for img_file in listdir(root_folder + '/' + episode):
            if img_file.endswith(".csv"):
                #read labels from csv file 
                data = pd.read_csv(img_file)
                reward = np.array(data.values).astype(int)
                rewards.append(reward)
            else:
                img = Image.open(root_folder + '/' + episode +'/'+ img_file)
                img_mat = np.array(img)
                images_per_episode.append(img_mat.flatten())
        images_per_episode = np.array(images_per_episode)
        pca = PCA(n_components=50)
        processed_imgs = pca.fit_transform(images_per_episode)
        listPCAimages.append(processed_imgs)
        break

    return listPCAimages,rewards

def get_training_data(listPCAimages,rewards):
    feature_data = []
    label_data = []

    for images,reward in zip(listPCAimages,rewards):
        for i in range(len(images)):
            img_sampling = np.array(random.sample(images[i:i+7], 5))
            vec_img = img_sampling.flatten()

            #create training data
            feature_data.append(vec_img)
            label_data.append(reward[i+9])

    return feature_data,label_data

listPCAimages,rewards = preprocess_img('/home/gaurav/Desktop/IITD_1stsem/ML/train_dataset')
feature_data,label_data = get_training_data(listPCAimages,rewards)
print(feature_data)
print(label_data)


