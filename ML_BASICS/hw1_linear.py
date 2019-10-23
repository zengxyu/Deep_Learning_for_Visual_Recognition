# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np
import matplotlib.pyplot as plt


def predict(X, W, b):
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(np.dot(X, W) + b)


def sigmoid(a):
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    c = 1 / (1 + np.exp(-a))
    return c


def l2loss(X, y, W, b):
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    sigma = predict(X, W, b)
    loss = np.mean(np.power((y - sigma), 2))
    w_grad = -2 * np.dot((y - sigma), np.dot(np.dot(sigma, (1 - sigma)), X)) / len(y)
    b_grad = -2 * np.mean(np.dot((y - sigma), np.dot(sigma, (1 - sigma))))

    # acc = np.mean(np.round(sigma) == y)
    # print "acc:", acc
    return loss, w_grad, b_grad


def train(X, y, W, b, num_iters=1000, eta=0.001):
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
    loss_list = []
    for i in range(num_iters):
        loss, w_grad, b_grad = l2loss(X, y, W, b)
        W = W - w_grad * eta
        b = b - b_grad * eta
        loss_list.append(loss)
        if i % 100 == 0:
            print "iteration:", i, ";   loss:", loss
    # Plot the loss function for t = 1000 iterations
    plot_loss(range(num_iters), loss_list)

    return W, b


def plot_loss(keys, loss_list):
    plt.plot(keys, loss_list, "*-", color="red")
    plt.xlabel("Iteration number")
    plt.ylabel("Loss value")
    plt.title("Loss Values for t = 1000 iterations")
    plt.show()
