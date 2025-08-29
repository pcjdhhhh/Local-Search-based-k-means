# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tool import *

def kmeans_plus_pair_small(dataset,k,rdx,rand_):
    
    n = dataset.shape[0]  
    
    new_add_index = rdx[0]
    centers = dataset[rdx]  
    
    cal_dis_num = 0  
    current_dis = np.array(np.ones(n)*np.inf)   
    current_index = np.array(np.ones(n)*0)      
    for temp_k in range(1,k+1):   
        
        tot = 0   
        for i,point in enumerate(dataset):
            #centers[len(centers)-1,:]
            current_dis[i],current_index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,current_dis[i],current_index[i])
            cal_dis_num = cal_dis_num+1
            
            tot = tot + current_dis[i]
        
        tot = tot * rand_[temp_k-1]   
        for i, di in enumerate(current_dis):
            tot -= di
            if tot > 0:
                continue
            if temp_k!=k:    
                centers=np.vstack((centers,dataset[i,:]))
            break
    return centers,cal_dis_num,current_dis,tot