# -*- coding: utf-8 -*-
from local_search_modal import *
from data import *
import time

file_name_save = ['coil100']

for file_name in file_name_save:
    print(file_name)
    data = get_coil100(file_name)
    num_of_pro = 2048 
    print(data.shape)
    
    rdx_name = 'save/' + file_name + '_' + 'rdx_' + str(num_of_pro)
    rand_name = 'save/' + file_name + '_' + 'rand_' + str(num_of_pro)
    sample_p_name = 'save/' + file_name + '_' + 'sample_p_' + str(num_of_pro)
    
    
    
        
        
    if os.path.exists(rdx_name):
        print('exist')
        rdx = np.array([int(np.loadtxt(rdx_name))])
        rand_ = np.loadtxt(rand_name)
        sample_p = np.loadtxt(sample_p_name)
    else:
        print('no exist')
        n = data.shape[0]  
        rdx = np.random.choice(range(n), 1)   
        rand_ = list()   
        for i in range(num_of_pro):
            rand_.append(random.random())
        np.savetxt(rdx_name,rdx)
        np.savetxt(rand_name,rand_)
        sample_p = list()   
        for i in range(num_of_pro):
            sample_p.append(random.random())
        np.savetxt(sample_p_name,sample_p)
        
            
        
        
    k = 16
    Z = 200
    lsds_uniformly_sample_name = 'save/' + file_name + '_' + 'lsds_uniformly_sample_' + str(k)
    if os.path.exists(lsds_uniformly_sample_name):
        print('exist')
        lsds_uniformly_sample = np.loadtxt(lsds_uniformly_sample_name).astype('int')
        #lsds_uniformly_sample = np.loadtxt(rdx_name))])
    else:
        print('no exist')
        lsds_uniformly_sample = np.random.choice(range(k), 2048)
        np.savetxt(lsds_uniformly_sample_name,lsds_uniformly_sample)
    
    
    
    print("----------------LSDS++ start-------------")
    s = time.time()
    C_LSDS_half,cost_LSDS_half,num_of_change_half,num_of_dis = LSDS_plus_LB_half(data,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample)
    e = time.time()
    print('LSDS_half time: ',e-s)
    print('changes: ',num_of_change_half)
    print('num_of_dis: ',num_of_dis)
    
    s = time.time()
    C_LSDS_mean,cost_LSDS_mean,num_of_change_LSDS,num_of_dis = LSDS_plus_LB_mean(data,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample,block_size=16)
    e = time.time()
    print('LSDS_mean time: ',e-s)
    print('changes: ',num_of_change_LSDS)
    print('num_of_dis: ',num_of_dis)
    
    s = time.time()
    C_LSDS_improved,cost_LSDS_improved,num_of_change_LSDS,num_of_dis = LSDS_plus_improved(data,k,Z,rdx,rand_,sample_p,lsds_uniformly_sample)
    e = time.time()
    print('LSDS_improved time: ',e-s)
    print('changes: ',num_of_change_LSDS)
    print('num_of_dis: ',num_of_dis)
    
    
    
    
    print('--------------------------------------------------------')
    
    
    print("----------------LS++ start-------------")
    
    
   
    
    s = time.time()
    C_LB_half,cost_LB_half,num_of_dis_half,num_of_change = LS_plus_lower_bound_half(data,k,Z,rdx,rand_,sample_p)
    e = time.time()
    print('LB_half time: ',e-s)
    print('distances computations: ',num_of_dis_half)
    print('num_of_change: ',num_of_change)
    
    
    s = time.time()
    C_LB_mean,cost_LB_mean,num_of_dis_mean,num_of_change = LS_plus_lower_bound_mean(data,k,Z,rdx,rand_,sample_p,block_size=16)
    e = time.time()
    print('LB_mean time: ',e-s)
    print('distances computations: ',num_of_dis_mean)
    print('num_of_change: ',num_of_change)
    
    
    
    
    s = time.time()
    C_improved_v2,cost_improved_v2,num_of_dis_v2,num_of_change = LS_plus_improved_v2(data,k,Z,rdx,rand_,sample_p)
    e = time.time()
    print('improved time: ',e-s)
    print('distances computations: ',num_of_dis_v2)
    
    
    
    
    
   
    print('--------------------------------------------------------')
    
    



