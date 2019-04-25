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

#set path of libsvm
sys.path.insert(0,"/home/gauravshukla789/machine_learning/libsvm-3.23/python")

from svm import svm_parameter, svm_problem
from svmutil import svm_train, svm_predict

def dump_object(dump_file,obj):
    writer = open(dump_file, 'wb')
    cPickle.dump(obj, writer)
    writer.close()

def get_object(dump_file):
    reader = open(dump_file, 'rb')
    obj = cPickle.load(reader)
    reader.close()
    return obj

def get_rewards(file_path):
    #read labels from csv file 
    data = pd.read_csv(file_path,header = None)
    reward = np.array(data.values).astype(int)

    #append one more reward for first image
    reward = np.vstack((np.array([[0]]),reward)).T[0]
    return reward

def preprocess_img(root_folder,dump_file,training_size):
    if os.path.exists(dump_file):
        return

    rewards = []
    images_per_50_episode = []
    list_episodes = sorted(listdir(root_folder))
    #process episodes for taining PCA
    for episode in list_episodes[:training_size]:
        print("Episode no:",episode)
    
        for img_file in listdir(root_folder + '/' + episode):
            if not img_file.endswith(".csv"):
                img = Image.open(root_folder + '/' + episode +'/'+ img_file).convert('LA')
                img_mat = np.array(img)
                images_per_50_episode.append(img_mat.flatten())

    images_per_50_episode = np.array(images_per_50_episode)
    pca = PCA(n_components=50)
    pca.fit(images_per_50_episode)

    listPCAimages = []
    #process episodes for training PCA
    for episode in list_episodes:
        print("Episode no:",episode)
        processed_imgs = []

        for img_file in listdir(root_folder + '/' + episode):
            if img_file.endswith(".csv"):
                #get rewards of this episode
                reward = get_rewards(root_folder + '/' + episode +'/'+ img_file)
                rewards.append(reward)
            else:
                img = Image.open(root_folder + '/' + episode +'/'+ img_file).convert('LA')
                img_mat = np.array(img)
                processed_imgs.append(pca.transform(img_mat.flatten()))
        listPCAimages.append(processed_imgs)
    #stripe out set of images episode wise
    rewards = np.array(rewards)
    
    dump_object(dump_file,[listPCAimages,rewards])

def get_training_data(dump_file,stride,samples):
    #read pickle file
    processed_data = get_object(dump_file)
    listPCAimages,rewards = processed_data[0],processed_data[1]

    feature_data = []
    label_data = []

    count = 0
    for images,reward in zip(listPCAimages,rewards):
        print(reward.shape,":",count)
        count += 1

        i = 0
        while True:
            #if i exceeds total length
            if(i+7 >= len(images)):
                break

            img_set = images[i:i+6]
            #get all combinations of images
            combs = np.array(list(itertools.combinations(img_set, 4)))
            length = len(combs)
            #no of samples based on current reward
            curr_reward = reward[i+7]
            if curr_reward == 1:
                rand_ind = np.random.choice(length, samples,replace=False)
            else:
                rand_ind = np.random.choice(length, 1,replace=False)      

            combs = combs[rand_ind]
            for comb in combs:
                img_sampling = np.vstack((np.array(comb),images[i+6]))
                vec_img = img_sampling.flatten()

                #create training data
                feature_data.append(vec_img)
                label_data.append(curr_reward)

            i += stride

    return feature_data,label_data

def train_data_csv(root_folder,dump_file,csv_file,stride,samples,training_size=50):

    if os.path.exists(csv_file):
        return

    preprocess_img(root_folder,dump_file,training_size)
    feature_data,label_data = get_training_data(dump_file,stride,samples)

    feature_data = np.array(feature_data)
    label_data = np.array(label_data)
    label_data = np.reshape(label_data, (label_data.shape[0],1)) 

    print(feature_data.shape)
    print(label_data.shape)

    #merge feature and label data
    train_data = np.hstack((feature_data,label_data))

    with open(csv_file , 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(train_data)
        print('Done')
    writeFile.close()

def get_data_from_csv(data_file):
    training_data = pd.read_csv(data_file, header = None)
    training_data = np.array(training_data.values)

    features = training_data[:,:-1]
    labels = training_data[:,-1]
    
    return features,labels

def lib_svm(train_file,test_file,kernel):
        print("inside libsvm")
        features, labels = get_data_from_csv(train_file)
        print(features)
        
        training_data = svm_problem(labels, features)
        
        if(kernel == 'gaussian'):
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
        else:
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.001275')
            
        model = svm_train(training_data, params)
        
        test_features, test_labels = get_data_from_csv(test_file)
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)
        return p_labels, p_acc, p_vals

def get_f1score_macro(prediction,original):
    return f1_score(original,prediction,average='macro')

if __name__ == '__main__':
    #root path
    path = '/home/gauravshukla789/machine_learning/train_dataset'

    #mode defines the consideration of all combinations, possible values={0,1}
    train_data_csv(path,"processed_data.pkl",path+'/train_data.csv',stride=1,samples=5,training_size=50)
    p_labels, p_acc, p_vals = lib_svm(path+'/train_data.csv', path+'/train_data.csv','linear')
    


