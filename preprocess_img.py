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
sys.path.insert(0,"/home/gaurav/Desktop/MachineLearning/libsvm-3.23")

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

    dump_object(dump_file,[listPCAimages,rewards])

def get_training_data(dump_file,mode):
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
            
            if(mode == 1):
                img_set = images[i:i+6]

                #get all combinations of images
                combs = list(itertools.combinations(img_set, 4))
                for comb in combs:
                    img_sampling = np.vstack((np.array(comb),images[i+7]))
                    vec_img = img_sampling.flatten()

                    #create training data
                    feature_data.append(vec_img)
                    label_data.append(reward[i+9])
            else:
                img_set = images[i:i+7]
                img_sampling = img_set[np.random.choice(img_set.shape[0], 5, replace=False), :]
                vec_img = img_sampling.flatten()

                #create training data
                feature_data.append(vec_img)
                label_data.append(reward[i+9])
    return feature_data,label_data

def train_data_csv(root_folder,dump_file,csv_file,mode):

    if os.path.exists(root_folder + '/' + csv_file):
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

    with open(root_folder + '/' + csv_file , 'w') as writeFile:
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
        features, labels = get_data_from_csv(train_file)
        training_data = svm_problem(labels, features)
        
        if(kernel == 'gaussian'):
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
        else:
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.001275')
            
        model = svm_train(training_data, params)
        
        test_features, test_labels = get_data_from_csv(test_file)
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)

if __name__ == '__main__':
    #reading the data from files
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    path = '/home/gaurav/Desktop/Machine Learning/train_dataset'

    #mode defines the consideration of all combinations, possible values={0,1}
    train_data_csv(path,"processed_data.pkl","train_data.csv",mode=1)
    lib_svm(path+'/train_data.csv', path+'/train_data.csv','linear')

