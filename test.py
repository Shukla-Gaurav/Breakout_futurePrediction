import numpy as np
import random
import itertools
# x=np.array([[1,2],[3,4]])
# print(x.flatten())

# episode_sizes = [6,3,1]
# processed_imgs = [3,4,5,1,2,33,22,65,78,88]
# listPCAimages = []
# Start_ind = 0
# for i in range(len(episode_sizes)):
#     listPCAimages.append(processed_imgs[Start_ind:Start_ind+episode_sizes[i]])
#     Start_ind += episode_sizes[i]
# print(listPCAimages)
# listPCAimages = np.array(listPCAimages)
# selected = [1,0]
# print(listPCAimages[selected])


img_set = [[1,2],[3,4],[5,6],[3,8],[4,4]]
#get all combinations of images
combs = np.array(list(itertools.combinations(img_set, 4)))
print(combs[:-1])

#no of samples based on current reward
curr_reward = 1
length = len(combs)

if curr_reward == 1:
    rand_ind = np.random.choice(length, 3,replace=False)
    combs = combs[rand_ind]
    
else:
    rand_ind = np.random.choice(length, 1,replace=False)
# print(rand_ind)  
# print(combs)