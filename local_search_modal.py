# -*- coding: utf-8 -*-

from k_means_plus_plus_modal import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tool import *
from preprocessing import *

def sample(dataset,current_dis,pp):
    n = dataset.shape[0] 
    
    tot = sum(current_dis)   
    tot = tot*pp  
    index = -1
    for i, di in enumerate(current_dis):
        tot -= di
        if tot > 0:
            continue
        p = dataset[i,:]    
        index = i   
        break
    return p,index
def improved_cost_lower_bound_mean(dataset,mean_info,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis,block_size):
    
   
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    cost = 0
    new_current_dis = current_dis.copy() 
    new_second_dis = second_dis.copy()
    new_current_index = current_index.copy()
    new_second_index = second_index.copy()
    new_C = C.copy()
    new_C[j,:] = p
    p_mean = mean_info[new_index,:]
    for i in range(n_data):
        query = dataset[i,:]
        query_mean = mean_info[i,:]
        
        #temp_dis = calEuclidean(query, p)
        lower_bound = calEuclidean(query_mean,p_mean) * block_size
        
        if new_current_index[i]==j:
            
            
            if lower_bound>new_second_dis[i]:
                
                new_current_index[i] = new_second_index[i]
                new_current_dis[i] = new_second_dis[i]
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
            
                if temp_dis<new_second_dis[i]:
                    
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
                else:
                    
                    new_current_index[i] = new_second_index[i]
                    new_current_dis[i] = new_second_dis[i]
                
        else:
            
            if lower_bound>new_current_dis[i]:
                pass
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
                if temp_dis<new_current_dis[i]:
                   
               
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
    return sum(new_current_dis),num_of_dis

def improved_cost_lower_bound_half(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis):
    
    
    [n_data,d] = dataset.shape
    n_C = C.shape[0]
    cost = 0
    new_current_dis = current_dis.copy() 
    new_second_dis = second_dis.copy()
    new_current_index = current_index.copy()
    new_second_index = second_index.copy()
    new_C = C.copy()
    new_C[j,:] = p
    
    for i in range(n_data):
        query = dataset[i,:]
        
        
        lower_bound = calEuclidean(query[0:int(d/2)],p[0:int(d/2)])
        
        if new_current_index[i]==j:
            
            
            if lower_bound>new_second_dis[i]:
                
                new_current_index[i] = new_second_index[i]
                new_current_dis[i] = new_second_dis[i]
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
            
                if temp_dis<new_second_dis[i]:
                    
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
                else:
                    
                    new_current_index[i] = new_second_index[i]
                    new_current_dis[i] = new_second_dis[i]
                
        else:
            
            if lower_bound>new_current_dis[i]:
                pass
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
                if temp_dis<new_current_dis[i]:
                   
                    
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
    return sum(new_current_dis),num_of_dis

def improved_cost_lower_bound_TCS(dataset,block_info,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis,block_size):
    
    
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    cost = 0
    new_current_dis = current_dis.copy() 
    new_second_dis = second_dis.copy()
    new_current_index = current_index.copy()
    new_second_index = second_index.copy()
    new_C = C.copy()
    new_C[j,:] = p
    p_mean = block_info[new_index,:]
    for i in range(n_data):
        query = dataset[i,:]
        query_mean = block_info[i,:]
        
        lower_bound = calEuclidean(query_mean,p_mean) 
        
        if new_current_index[i]==j:
            
            
            if lower_bound>new_second_dis[i]:
                
                new_current_index[i] = new_second_index[i]
                new_current_dis[i] = new_second_dis[i]
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
            
                if temp_dis<new_second_dis[i]:
                    
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
                else:
                    
                    new_current_index[i] = new_second_index[i]
                    new_current_dis[i] = new_second_dis[i]
                
        else:
            
            if lower_bound>new_current_dis[i]:
                pass
            else:
                temp_dis = calEuclidean(query, p)
                num_of_dis = num_of_dis + 1
                if temp_dis<new_current_dis[i]:
                    
                    
                    new_current_index[i] = j
                    new_current_dis[i] = temp_dis
    return sum(new_current_dis),num_of_dis



def improved_cost(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis):
    
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    cost = 0
    new_current_dis = current_dis.copy() 
    new_second_dis = second_dis.copy()
    new_current_index = current_index.copy()
    new_second_index = second_index.copy()
    new_C = C.copy()
    new_C[j,:] = p
    for i in range(n_data):
        query = dataset[i,:]
        
        temp_dis = calEuclidean(query, p)
        num_of_dis = num_of_dis + 1
        
        if new_current_index[i]==j:
            if temp_dis<new_second_dis[i]:
                
                new_current_index[i] = j
                new_current_dis[i] = temp_dis
            else:
                
                new_current_index[i] = new_second_index[i]
                new_current_dis[i] = new_second_dis[i]
                
                
        else:
            
            if temp_dis<new_current_dis[i]:
                
                
                new_current_index[i] = j
                new_current_dis[i] = temp_dis
    return sum(new_current_dis),num_of_dis

def improved_cost_update(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index):
    
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    cost = 0
    new_current_dis = current_dis.copy() 
    new_second_dis = second_dis.copy()
    new_current_index = current_index.copy()
    new_second_index = second_index.copy()
    new_C = C.copy()
    new_C[j,:] = p
    for i in range(n_data):
        query = dataset[i,:]
        
        temp_dis = calEuclidean(query, p)
       
        if new_current_index[i]==j:
            
            if temp_dis<new_second_dis[i]:
                
                
                new_current_index[i] = j
                new_current_dis[i] = temp_dis
               
            else:

                dis_save = np.array([calEuclidean(query, b) for b in new_C])
                sort_index = np.argsort(dis_save)
                new_current_index[i] = sort_index[0]
                new_current_dis[i] = dis_save[sort_index[0]]
                new_second_index[i] = sort_index[1]
                new_second_dis[i] = dis_save[sort_index[1]]
            
        else:
           
            if temp_dis<new_current_dis[i]:
                
                new_second_dis[i] = new_current_dis[i]
                new_second_index[i] = new_current_index[i]
                new_current_index[i] = j
                new_current_dis[i] = temp_dis
                
            elif temp_dis>new_second_dis[i]:
                
                a = new_current_index[i]
                b = new_second_index[i]
                
                dis_save = np.array([calEuclidean(query, b) for b in new_C])
                sort_index = np.argsort(dis_save)
                new_current_index[i] = sort_index[0]
                new_current_dis[i] = dis_save[sort_index[0]]
                new_second_index[i] = sort_index[1]
                new_second_dis[i] = dis_save[sort_index[1]]
                
            else:
                
                new_second_dis[i] = temp_dis
                new_second_index[i] = j
                
    return new_current_dis,new_current_index,new_second_dis,new_second_index
   
def naive_cost(dataset,C):
    
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    cost = 0
    current_dis = np.array(np.ones(n_data)*np.inf)   
    for i in range(n_data):
        query = dataset[i,:]
        min_ = np.inf
        for j in range(n_C):
            c = C[j,:]
            temp_dis = calEuclidean(query,c)
            if temp_dis<min_:
                min_ = temp_dis
        current_dis[i] = min_
        cost = cost + min_
    return cost,current_dis

def get_first_second_points(dataset,C):
   
    n_data = dataset.shape[0]
    n_C = C.shape[0]
    current_dis = np.array(np.ones(n_data)*np.inf)   
    current_index = np.array(np.ones(n_data)-2)   
    second_dis = np.array(np.ones(n_data)*np.inf)
    second_index = np.array(np.ones(n_data)-2)
    
    for i in range(n_data):
        query = dataset[i,:]
        current_min = np.inf
        second_min = np.inf
        temp_dis = np.array([calEuclidean(query,c) for c in C])
        
        sort_index = np.argsort(temp_dis)
        #print(temp_dis)
        current_index[i] = sort_index[0]
        current_dis[i] = temp_dis[sort_index[0]]
        second_index[i] = sort_index[1]
        second_dis[i] = temp_dis[sort_index[1]]
    return current_dis,current_index,second_dis,second_index
       
def LS_plus_lower_bound_half(dataset,k,Z,rdx,rand_,sample_p):
    
    res_cost = np.zeros(Z)
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)  
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    min_index = -1     
    change = 0
    #mean_info = preprocessing_with_mean(dataset, block_size)
    num_of_dis = 0
    num_of_change = 0
    
    for i in range(Z):
        
        p,new_index = sample(dataset,current_dis,sample_p[i])
       
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        min_cost = np.inf
        min_index = -1
        
        for j in range(k):
            temp_cost,num_of_dis = improved_cost_lower_bound_half(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis)
            if temp_cost<min_cost:
                min_cost = temp_cost
                min_index = j
        change = 0
       
        if min_cost<Z_cost:
            change = 1
            C[min_index] = p
        if change==1:
            #print('ok')
            num_of_change = num_of_change + 1
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            #current_dis,current_index,second_dis,second_index = improved_cost_update(dataset,C,p,new_index,min_index,current_dis,current_index,second_dis,second_index)
            num_of_dis = num_of_dis + k*dataset.shape[0]
    return C,res_cost,num_of_dis,num_of_change
        
def LS_plus_lower_bound_mean(dataset,k,Z,rdx,rand_,sample_p,block_size):
   
    res_cost = np.zeros(Z)
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)  
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    min_index = -1     
    change = 0
    mean_info = preprocessing_with_mean(dataset, block_size)
    num_of_dis = 0
    num_of_change = 0
    
    for i in range(Z):
       
        p,new_index = sample(dataset,current_dis,sample_p[i])
        #print(new_index)
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        min_cost = np.inf
        min_index = -1
        #print(sum(current_dis))
        for j in range(k):
            temp_cost,num_of_dis = improved_cost_lower_bound_mean(dataset,mean_info,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis,block_size)
            if temp_cost<min_cost:
                min_cost = temp_cost
                min_index = j
        change = 0
        #print('min cost: ',min_cost,' Z_cost: ',Z_cost)
        if min_cost<Z_cost:
            change = 1
            C[min_index] = p
        if change==1:
            #print('ok')
            num_of_change = num_of_change + 1
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            #current_dis,current_index,second_dis,second_index = improved_cost_update(dataset,C,p,new_index,min_index,current_dis,current_index,second_dis,second_index)
            num_of_dis = num_of_dis + k*dataset.shape[0]
    return C,res_cost,num_of_dis,num_of_change


    
def LS_plus_improved_v2(dataset,k,Z,rdx,rand_,sample_p):
    
    res_cost = np.zeros(Z)
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)  
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    min_index = -1     
    change = 0
    num_of_dis = 0
    num_of_change = 0
    for i in range(Z):
        
        p,new_index = sample(dataset,current_dis,sample_p[i])
        #print(new_index)
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        min_cost = np.inf
        min_index = -1
        #print(sum(current_dis))
        for j in range(k):
            temp_cost,num_of_dis = improved_cost(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis)
            if temp_cost<min_cost:
                min_cost = temp_cost
                min_index = j
        change = 0
        #print('min cost: ',min_cost,' Z_cost: ',Z_cost)
        if min_cost<Z_cost:
            change = 1
            C[min_index] = p
        if change==1:
            num_of_change = num_of_change + 1
            #current_dis,current_index,second_dis,second_index = improved_cost_update(dataset,C,p,new_index,min_index,current_dis,current_index,second_dis,second_index)
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            num_of_dis = num_of_dis + k*dataset.shape[0]
            #print(sum(current_dis))
    return C,res_cost,num_of_dis,num_of_change


def LSDS_plus_LB_half(dataset,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample):
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心
    res_cost = np.zeros(Z)
    num_of_dis = 0
    num_of_change = 0
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    
    for i in range(Z):
        p,new_index = sample(dataset,current_dis,sample_p[i])
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        
        dis_save = np.array([calEuclidean(p, b) for b in C])
        
        sort_index = np.argsort(dis_save)
        q_bar_index = sort_index[0]
        q_bar = C[q_bar_index,:]
        
        
        q_index = lsds_uniformly_sample[i]
        q = C[q_index,:]
        q_change_cost,num_of_dis = improved_cost_lower_bound_half(dataset,C,p,new_index,q_index,current_dis,current_index,second_dis,second_index,num_of_dis)
        #improved_cost_lower_bound_half(dataset,C,p,new_index,q_bar_index,current_dis,current_index,second_dis,second_index,num_of_dis)
        q_bar_change_cost,num_of_dis = improved_cost_lower_bound_half(dataset,C,p,new_index,q_bar_index,current_dis,current_index,second_dis,second_index,num_of_dis)
        
        change_q_q_bar = 0
        if q_change_cost>q_bar_change_cost:
            change_q_q_bar = 1
            q = q_bar
            q_index = q_bar_index
        if change_q_q_bar==1:   
            min_cost = q_bar_change_cost
        else:
            min_cost = q_change_cost
        change = 0
        if min_cost<Z_cost:
            change = 1
            num_of_change = num_of_change + 1
            C[q_index] = p
        if change==1:
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            num_of_dis = num_of_dis + k*dataset.shape[0]
    return C,res_cost,num_of_change,num_of_dis

def LSDS_plus_LB_mean(dataset,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample,block_size):
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   
    res_cost = np.zeros(Z)
    num_of_dis = 0
    num_of_change = 0
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    mean_info = preprocessing_with_mean(dataset, block_size)
    for i in range(Z):
        p,new_index = sample(dataset,current_dis,sample_p[i])
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        
        dis_save = np.array([calEuclidean(p, b) for b in C])
        
        sort_index = np.argsort(dis_save)
        q_bar_index = sort_index[0]
        q_bar = C[q_bar_index,:]
        
        
        q_index = lsds_uniformly_sample[i]
        q = C[q_index,:]
        q_change_cost,num_of_dis = improved_cost_lower_bound_mean(dataset,mean_info,C,p,new_index,q_index,current_dis,current_index,second_dis,second_index,num_of_dis,block_size)
        #improved_cost_lower_bound_mean(dataset,mean_info,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis,block_size=16)
        q_bar_change_cost,num_of_dis = improved_cost_lower_bound_mean(dataset,mean_info,C,p,new_index,q_bar_index,current_dis,current_index,second_dis,second_index,num_of_dis,block_size)
        
        change_q_q_bar = 0
        if q_change_cost>q_bar_change_cost:
            change_q_q_bar = 1
            q = q_bar
            q_index = q_bar_index
        if change_q_q_bar==1:   
            min_cost = q_bar_change_cost
        else:
            min_cost = q_change_cost
        change = 0
        if min_cost<Z_cost:
            change = 1
            num_of_change = num_of_change + 1
            C[q_index] = p
        if change==1:
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            num_of_dis = num_of_dis + k*dataset.shape[0]
    return C,res_cost,num_of_change,num_of_dis



def LSDS_plus_improved(dataset,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample):
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心
    res_cost = np.zeros(Z)
    num_of_dis = 0
    num_of_change = 0
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    for i in range(Z):
        p,new_index = sample(dataset,current_dis,sample_p[i])
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        
        dis_save = np.array([calEuclidean(p, b) for b in C])
        sort_index = np.argsort(dis_save)
        q_bar_index = sort_index[0]
        q_bar = C[q_bar_index,:]
        
        
        q_index = lsds_uniformly_sample[i]
        q = C[q_index,:]
        q_change_cost,num_of_dis = improved_cost(dataset,C,p,new_index,q_index,current_dis,current_index,second_dis,second_index,num_of_dis)
        
        q_bar_change_cost,num_of_dis = improved_cost(dataset,C,p,new_index,q_bar_index,current_dis,current_index,second_dis,second_index,num_of_dis)
        
        change_q_q_bar = 0
        if q_change_cost>q_bar_change_cost:
            change_q_q_bar = 1
            q = q_bar
            q_index = q_bar_index
        if change_q_q_bar==1:   
            min_cost = q_bar_change_cost
        else:
            min_cost = q_change_cost
        change = 0
        if min_cost<Z_cost:
            change = 1
            num_of_change = num_of_change + 1
            C[q_index] = p
        if change==1:
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            num_of_dis = num_of_dis + k*dataset.shape[0]
    return C,res_cost,num_of_change,num_of_dis
        
        

def LSDS_plus_naive(dataset,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample):
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心
    res_cost = np.zeros(Z)
    num_of_dis = 0
    num_of_change = 0
    
    for i in range(Z):
        
        p,new_index = sample(dataset,current_dis,sample_p[i])
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        
        
        dis_save = np.array([calEuclidean(p, b) for b in C])
        sort_index = np.argsort(dis_save)
        q_bar_index = sort_index[0]
        q_bar = C[q_bar_index,:]
        
       
        q_index = lsds_uniformly_sample[i]
        q = C[q_index,:]
        
        
        C_copy = C.copy()
        C_copy[q_index,:] = p
        q_change_cost,_ = naive_cost(dataset,C_copy)
        
        C_copy = C.copy()
        C_copy[q_bar_index,:] = p
        q_bar_change_cost,_ = naive_cost(dataset,C_copy)
        
        change_q_q_bar = 0
        if q_change_cost>q_bar_change_cost:
            change_q_q_bar = 1
            q = q_bar
            q_index = q_bar_index
        if change_q_q_bar==1:   
            min_cost = q_bar_change_cost
        else:
            min_cost = q_change_cost
        
        if min_cost<Z_cost:
            num_of_change = num_of_change + 1
            C[q_index] = p
        Z_cost,current_dis = naive_cost(dataset,C)
    return C,res_cost,num_of_change
            
        
        
        
        

def LS_plus_improved(dataset,k,Z,rdx,rand_,sample_p):
    
    res_cost = np.zeros(Z)
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心 注意这里的kmeans_plus_pair_small函数发生变化（与kmeans++比较）
                                                                          
    current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
    num_of_dis = 0
    num_of_change = 0
    for i in range(Z):
       
        #print(i)
        p,new_index = sample(dataset,current_dis,sample_p[i])
        #print(new_index)
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        min_cost = np.inf
        min_index = -1
        #print(sum(current_dis))
        for j in range(k):
            temp_cost,num_of_dis = improved_cost(dataset,C,p,new_index,j,current_dis,current_index,second_dis,second_index,num_of_dis)
            if temp_cost<min_cost:
                min_cost = temp_cost
                min_index = j
        #print(min_cost)
        change = 0
        #print('min cost: ',min_cost,' Z_cost: ',Z_cost)
        if min_cost<Z_cost:
            #print('change')
            change=1
            C[min_index] = p
        if change==1:   
            current_dis,current_index,second_dis,second_index = get_first_second_points(dataset,C)
            num_of_dis = num_of_dis + k*dataset.shape[0]
            num_of_change = num_of_change + 1
    return C,res_cost,num_of_dis,num_of_change

def local_search_naive(dataset,k,Z,rdx,rand_,sample_p):
    
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心
    res_cost = np.zeros(Z)
    n_data = dataset.shape[0]
    for i in range(Z):
        
        print(i)
        Z_cost = sum(current_dis)
        min_cost = np.inf
        min_j_index = -1
        min_kk_index = -1
        res_cost[i] = Z_cost
        
        for j in range(n_data):
            for kk in range(k):
                print(j)
                C_copy = C.copy()
                C_copy[kk,:] = dataset[j,:]  
                temp_cost,_ = naive_cost(dataset,C_copy)
                if temp_cost<min_cost:
                    min_cost = temp_cost
                    min_j_index = j
                    min_kk_index = kk
        
        change = 0
        if min_cost<Z_cost:
            C[min_kk_index,:] = dataset[min_j_index,:]
        Z_cost,current_dis = naive_cost(dataset,C)
    return C,res_cost


    
    


def LS_plus_naive(dataset,k,Z,rdx,rand_,sample_p):
    
    C,_,current_dis,tot = kmeans_plus_pair_small(dataset,k,rdx,rand_)   #首先使用k_means++算法生成初始的聚类中心
    #print(sum(current_dis))
    res_cost = np.zeros(Z)
    for i in range(Z):
        
        p,new_index = sample(dataset,current_dis,sample_p[i])
        
        Z_cost = sum(current_dis)
        res_cost[i] = Z_cost
        min_cost = np.inf
        min_index = -1
        for j in range(k):
            C_copy = C.copy()
            C_copy[j,:] = p   
            temp_cost,_ = naive_cost(dataset,C_copy)
            if temp_cost<min_cost:
                min_cost = temp_cost
                min_index = j
        
        
        change = 0
        if min_cost<Z_cost:
            change = 1
            C[min_index] = p
        #if change==1:
        Z_cost,current_dis = naive_cost(dataset,C)
    return C,res_cost
            
        
        
        
    
    
    
    
    
    