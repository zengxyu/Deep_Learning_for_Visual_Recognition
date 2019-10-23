# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""

from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time


# Visualize the confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def task1b(X_train, y_train, X_test, y_test):
    # Randomly subset the training set , test the first 10 images, get the confusion matrix
    exm_num_per_class = 100
    k_set = [1, 5]
    X_training = []
    y_training = []
    for i in range(10):
        x_train_i = X_train[y_train == i]
        y_train_i = y_train[y_train == i]
        random_indexes = np.random.choice(range(len(y_train_i)), size=exm_num_per_class, replace=False)
        # random_indexes = np.random.randint(0, len(y_train_i), size=exm_num_per_class)
        x_train_i = x_train_i[random_indexes]
        y_train_i = y_train_i[random_indexes]
        X_training.extend(x_train_i.copy())
        y_training.extend(y_train_i.copy())
    X_testing = X_test[0:10]
    y_testing = y_test[0:10]
    # Test on test data
    for k in k_set:
        # 1) Compute distances:
        dists = mlBasics.compute_euclidean_distances(np.array(X_training), np.array(X_testing))
        # 2) Run the code below and predict labels:
        y_test_pred = mlBasics.predict_labels(dists, y_training, k=k)
        print 'For k = ', k, ' : {0:0.02f}'.format(
            np.mean(y_test_pred == y_testing) * 100), "of test examples classified correctly."
        cm_1 = confusion_matrix(y_testing, y_test_pred, labels=range(10))
        title = "Confusion Matrix k = " + str(k)
        plot_confusion_matrix(cm_1, range(10), title)


def plot_acc_for_k(mean_acc_record, keys):
    print "K = ", keys[np.argmax(mean_acc_record)], " perform best!"
    plt.plot(keys, mean_acc_record, "*-", color="red")
    plt.title("The accuracies for different keys")
    plt.xticks(keys)
    plt.xlabel("The number K for neighbors")
    plt.ylabel("Accuracy")
    plt.show()


def task1c(X_train, y_train):
    '''
    implement the 5-fold cross validation
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    '''
    # K hyper parameters need to be test
    k_set = range(1, 16)
    # n-fold
    fold_num = 5
    # example numbers per class, we need 100 images for one class, eg. class 1
    exm_num_per_class = 100
    # Initial the folds for training and folds for testing, and their corresponding label
    folds_training = [[] for i in range(fold_num)]
    label_for_folds_training = [[] for i in range(fold_num)]
    folds_testing = [[] for i in range(fold_num)]
    label_for_folds_testing = [[] for i in range(fold_num)]
    # For each class (number 1-10), sample 100 images randomly.
    for i in range(10):
        x_train_i = X_train[y_train == i]
        y_train_i = y_train[y_train == i]
        random_indexes = np.random.choice(range(len(y_train_i)), size=exm_num_per_class, replace=False)
        # random_indexes = np.random.randint(0, len(y_train_i), size=exm_num_per_class)
        x_train_i = x_train_i[random_indexes]
        y_train_i = y_train_i[random_indexes]
        stepsize = exm_num_per_class / fold_num
        for n in range(fold_num):
            # Use the one fold to test, and the rest to train
            folds_testing[n].append(x_train_i[stepsize * n:stepsize * (n + 1)])
            label_for_folds_testing[n].append(y_train_i[stepsize * n:stepsize * (n + 1)])
            folds_training[n].append(np.delete(x_train_i, range(stepsize * n, stepsize * (n + 1)), axis=0))
            label_for_folds_training[n].append(np.delete(y_train_i, range(stepsize * n, stepsize * (n + 1)), axis=0))
    # Variable result_records = {1: [acc1,...,acc5],2:[acc1,...,acc5]...} is to record the accuracies for each class
    result_records = {}
    for n in range(fold_num):
        # Following is to reshape
        folds_train = np.reshape(folds_training[n], (-1, np.shape(folds_training[n])[-1]))
        folds_test = np.reshape(folds_testing[n], (-1, np.shape(folds_testing[n])[-1]))
        label_for_folds_train = np.reshape(label_for_folds_training[n], -1)
        label_for_folds_test = np.reshape(label_for_folds_testing[n], -1)
        # Compute the distance
        dists = mlBasics.compute_euclidean_distances(folds_train, folds_test)
        # Iterate k
        for k in k_set:
            if k not in result_records.keys():
                result_records[k] = []
            y_test_pred = mlBasics.predict_labels(dists, label_for_folds_train, k=k)
            acc = np.mean(y_test_pred == label_for_folds_test) * 100
            # add the acc to the result_records
            result_records[k].append(acc)
            # print '{0:0.02f}'.format(acc), "of test examples classified correctly."
    mean_acc_record = [np.mean(result_records[key]) for key in result_records.keys()]
    plot_acc_for_k(mean_acc_record, result_records.keys())


def task1d(X_train, y_train, X_test, y_test):
    """
    Compare the computational cost of classifying the test data
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # k = 1 neighbor and the k neighbor which performed best in 1c question
    k_set = [1, 3]

    dists = mlBasics.compute_euclidean_distances(X_train, X_test)
    # The list of computational costs for two ks
    c_cost_lists = []
    acc_lists = []
    for k in k_set:
        start_time = time.time()
        y_test_pred = mlBasics.predict_labels(dists, y_train, k)
        duration_time = time.time() - start_time
        c_cost_lists.append(duration_time)
        acc = np.mean(y_test_pred == y_test) * 100
        acc_lists.append(acc)
    increased_time = c_cost_lists[1] - c_cost_lists[0]
    acc_my_classifier = acc_lists[1]
    print "The increased computation time is ", increased_time
    print "The accuracy of my classifier if ", acc_my_classifier


# print np.shape(folds),np.shape(label_for_folds)


# Load data - two class
# X_train, y_train = load_mnist('training' , [0,1] )
# X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')
# Get the  first ten images

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print "Question 1b:"
# task1b(X_train, y_train, X_test, y_test)
print "Question 1c:"
task1c(X_train, y_train)
# print "Question 1d:"
# task1d(X_train, y_train, X_test, y_test)
