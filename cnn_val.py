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

def dump_object(dump_file,obj):
    writer = gzip.open(dump_file, 'wb')
    cPickle.dump(obj, writer,-1)
    writer.close()

def get_rewards(file_path):
    #read labels from csv file 
    data = pd.read_csv(file_path,header = None)
    reward = np.array(data.values).astype(int)
    return reward

def preprocess_img(rewards, root_folder,pickle_folder,batch_size):

    list_episodes = sorted(listdir(root_folder))
    epi_count = 0 
    feature_data = []
    st_ind = 0
    #process episodes for training data
    for episode in list_episodes[:-1]:
        epi_count += 1
        images_per_episode = []
        print("Episode no:",episode)
        images_folder = sorted(listdir(root_folder + '/' + episode))

        for img_file in images_folder:
            if img_file.endswith(".png"):
                imagepil = Image.open(root_folder + '/' + episode +'/'+ img_file)
                imagepil = np.array(imagepil.convert('L')) 
                images_per_episode.append(imagepil)
                
        images_per_episode = np.array(images_per_episode)
        img_stack = np.stack(images_per_episode, axis = 2)
        feature_data.append(img_stack)

        #dump object for each batch
        if (len(feature_data) == batch_size) or (len(list_episodes)-1 == epi_count):
            dump_file = pickle_folder + '/' + str(episode) + '.pkl'
            #stripe out rewards
            label_data = rewards[st_ind:st_ind+batch_size]
            st_ind += batch_size

            dump_object(dump_file,[feature_data,label_data])
            feature_data = []

if __name__ == '__main__':
    #root path
    path = '/home/gaurav/Desktop/IITD_1stsem/ML/val_dataset'
    pickle_folder = sys.argv[1]
    val_csv = sys.argv[2]

    #get rewards from csv
    rewards = get_rewards(val_csv)

    #dump the train data per episode
    batch_size = 3000
    preprocess_img(rewards,path,pickle_folder,batch_size)
    
    


