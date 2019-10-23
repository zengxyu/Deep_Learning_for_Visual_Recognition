# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np


def compute_euclidean_distances(X, Y):
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    # testing data
    M = Y.shape[0]
    # training data
    N = X.shape[0]
    dist = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            dist[i, j] = np.sqrt(np.sum(np.square(Y[i] - X[j])))
    return dist


def predict_labels(dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    (M, N) = np.shape(dists)
    # M * k
    # M个测试数据，对每一个测试数据,前k个距离最近的图像
    pred_labels = np.zeros(M)
    for m in range(M):
        index_k = np.argsort(dists[m])[0:k]
        labels_k = [labels[index_k[j]] for j in range(k)]
        counts = {}
        for item in labels_k:
            if item in counts.keys():
                counts[item] += 1
            else:
                counts[item] = 1
        pred_labels[m] = max(counts, key=lambda key: counts[key])

    return pred_labels
