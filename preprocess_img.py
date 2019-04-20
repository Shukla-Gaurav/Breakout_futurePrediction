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

#set path of libsvm
sys.path.insert(0,"/home/gaurav/Desktop/MachineLearning/libsvm-3.23/python")

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

def preprocess_img(root_folder,dump_file):
    if os.path.exists(dump_file):
        return

    rewards = []
    episode_sizes = []
    images_per_50_episode = []
    count = 1

    #process only 50 episodes
    for episode in listdir(root_folder):
        if count > 50:
            break
        count += 1
        print(episode)

        #store size of each episode
        episode_sizes.append(len(listdir(root_folder + '/' + episode)))

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
                images_per_50_episode.append(img_mat.flatten())
    images_per_50_episode = np.array(images_per_50_episode)
    pca = PCA(n_components=50)
    processed_imgs = pca.fit_transform(images_per_50_episode)
    rewards = np.array(rewards)

    #stripe out set of images episode wise
    listPCAimages = []
    Start_ind = 0
    for i in range(len(episode_sizes)):
        listPCAimages.append(processed_imgs[Start_ind:Start_ind+episode_sizes[i]])
        Start_ind += episode_sizes[i]

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
            combs = list(itertools.combinations(img_set, 4))

            #no of samples based on current reward
            curr_reward = reward[i+7]
            if curr_reward == 1:
                combs = random.choices(combs, samples)
            else:
                combs = random.choices(combs, 1)          

            for comb in combs:
                img_sampling = np.vstack((np.array(comb),images[i+6]))
                vec_img = img_sampling.flatten()

                #create training data
                feature_data.append(vec_img)
                label_data.append(curr_reward)

            i += stride

    return feature_data,label_data

def train_data_csv(root_folder,dump_file,csv_file,mode):

    if os.path.exists(csv_file):
        return

    preprocess_img(root_folder,dump_file)
    feature_data,label_data = get_training_data(dump_file,mode)

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
        
        # training_data = svm_problem(labels, features)
        
        # if(kernel == 'gaussian'):
        #     params = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
        # else:
        #     params = svm_parameter('-s 0 -t 2 -c 1 -g 0.001275')
            
        # model = svm_train(training_data, params)
        
        # test_features, test_labels = get_data_from_csv(test_file)
        # p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)

if __name__ == '__main__':
    #root path
    path = '/home/gaurav/Desktop/MachineLearning/train_dataset'

    #mode defines the consideration of all combinations, possible values={0,1}
    train_data_csv(path,"processed_data.pkl",path+'/train_data.csv',mode=1)
    print(lib_svm(path+'/train_data.csv', path+'/train_data.csv','linear'))

