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

    #append one more reward for first image
    reward = np.vstack((np.array([[0]]),reward)).T[0]
    return reward

def dump_data_per_episode(dump_file,images,rewards,stride,samples):   
    feature_data = []
    label_data = []

    i = 0
    consume = 15 
    while True:
        #if i exceeds total length
        if(i+7 >= len(images)):
            break
        #get all index combinations of images
        ind_combs = np.array(list(itertools.combinations(np.arange(i,i+6), 4)))
        length = len(ind_combs)
        #no of samples based on current reward
        curr_reward = rewards[i+7]
        if curr_reward == 1:
            rand_ind = np.random.choice(length,samples,replace=False)
            consume += 2*samples
        else:
            rand_ind = np.random.choice(length, 1,replace=False)
            if consume > 0:
                consume = consume - 1   

        ind_combs = ind_combs[rand_ind]

        if consume > 0:
            for indexes in ind_combs:
                #append the last image
                indexes = np.append(indexes,i+6)
                img_sampling = np.array(images[indexes])
                img_stack = np.stack(img_sampling, axis = 2)
                #create training data
                feature_data.append(img_stack)
                label_data.append(curr_reward)

        i += stride
    print("1's:",np.sum(np.array(label_data)))
    print("total len:",len(label_data))
    dump_object(dump_file,[feature_data,label_data])

def preprocess_img(root_folder,pickle_folder,stride,samples):

    list_episodes = sorted(listdir(root_folder))
    
    #process episodes for training data
    for episode in list_episodes:
        images_per_episode = []
        rewards = []
        print("Episode no:",episode)
        #store size of each episode

        images_folder = sorted(listdir(root_folder + '/' + episode))

        for img_file in images_folder:
            if img_file.endswith(".csv"):
                #get rewards of this episode
                rewards = get_rewards(root_folder + '/' + episode +'/'+ img_file)
            elif img_file.endswith(".png"):
                imagepil = Image.open(root_folder + '/' + episode +'/'+ img_file)
                imagepil = np.array(imagepil.convert('L')) 
                images_per_episode.append(imagepil)
                
        images_per_episode = np.array(images_per_episode)
        #print(imagepil.shape)
        dump_file = root_folder + '/' + pickle_folder + '/' + str(episode) + '.pkl'
        dump_data_per_episode(dump_file,images_per_episode,rewards,stride,samples)

if __name__ == '__main__':
    #root path
    path = '/home/gaurav/Desktop/IITD_1stsem/ML/train_dataset'
    pickle_folder = sys.argv[1]
    #dump the train data per episode
    preprocess_img(path,pickle_folder,stride=1,samples=15)
    
    


