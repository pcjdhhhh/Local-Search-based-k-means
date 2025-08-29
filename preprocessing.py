# -*- coding: utf-8 -*-

import numpy as np
import math

def preprocessing_with_mean(train,block_size):
    [n_train,dim] = train.shape
    if dim%block_size == 0:
        
        block_num = dim // block_size  
    else:
        
        block_num = math.floor(dim / block_size) + 1
        should_add = block_size - (dim % block_size)
        #new_dim = dim + should_add
        add_train = np.zeros((n_train,should_add))
        train = np.hstack((train,add_train))  
    [n_train,dim] = train.shape
    mean_info = np.zeros((n_train,block_num))
    for i in range(n_train):
        current = train[i,:]
        #norm_info[i] = np.linalg.norm(current)**2
        for j in range(block_num):
            s = j*block_size
            e = (j+1)*block_size
            mean_info[i,j] = np.mean(current[s:e])
    return mean_info