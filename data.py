# -*- coding: utf-8 -*-

import pandas

from scipy.io import loadmat
import numpy as np

def get_coil100():
    path = 'datasets/coil100/COIL100.mat'
    data = loadmat(path)
    return data['fea'].astype(float)




