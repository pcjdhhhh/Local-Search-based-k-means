# -*- coding: utf-8 -*-


def calEuclidean(a,b):
    
    length = len(a)
    sum_ = 0
    for i in range(length):
        sum_ = sum_ + (a[i]-b[i])**2
    return sum_

def get_nearest_dist_using_pair_small(point,new_center,new_index,current_dis,current_index):
   
    temp_dis = calEuclidean(point,new_center)
    if temp_dis<current_dis:
       
        current_dis = temp_dis
        current_index = new_index
    return current_dis,current_index


