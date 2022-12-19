import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn import linear_model
from sklearn.svm import SVC


def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)


def mean_centered(data):
    data_xy = data[:,:2]
    means = np.ones(data_xy.shape)
    ones = np.mean(data_xy, axis=0)

    # https://www.geeksforgeeks.org/numpy-column_stack-in-python/
    # did not want to transform data into 1d to use hstack
    data_n = np.column_stack((data_xy - means*ones, data[:,2]))
    return data_n


def get_proj1(data):
    proj1 = np.ones((len(data), 3))
    proj1[:,0:2] = proj1[:,0:2]*data[:,0:2]
    proj1[:,2] = proj1[:,2]*np.exp(-1*((proj1[:,0])**2+(proj1[:,1])**2))
    return proj1


def get_proj2(data):
    proj2 = np.ones((len(data), 3))
    proj2[:,0] = data[:,0]**2
    proj2[:,1] = (2**(1/2))*data[:,0]*data[:,0]
    proj2[:,2] = data[:,1]**2
    return proj2


def ten_fold(data_with_classes, kernel: str, mean_center_flag: bool):
    """
        :param data_with_classes: a given dataset
        :param kernel: a specified kernel(rbf, poly, linear)
        :param mean_center_flag: need to mean-center the data?
        :return:
    """

    # Set constants
    poly_kernel_degree = 2
    k = 10  # 10 folds
    if mean_center_flag:
        data_with_classes = mean_centered(data_with_classes)

    # Indexes to split data into inputs and outputs for predictions
    inds_in = [0, 1]
    inds_out = 2

    # Dividing the data into k sets
    sets = []
    size = len(data_with_classes) // k
    for i in range(1, k + 1):
        sets.append(data_with_classes[size * (i - 1): size * i])

    # # Computing train and test errors for where
    # # each set acts as the testing data
    # Initial Train and test error lists
    train_errors = []
    test_errors = []

    # Compute for each set
    for i in range(len(sets)):
        test_set = sets[i]
        # Avoid error with concatenating arrays of shape (0,0) and (len(sets),3)
        train_set = [[0, 0, 0]]
        for j in range(len(sets)):
            if j != i:
                train_set = np.concatenate((train_set, sets[j]))
        train_set = train_set[1:]

        # Fit model to training dataset
        # Maybe there's a better way to construct to make reusable
        if kernel == 'rbf':
            line_svm = SVC(kernel=kernel, C=0.000001)
        elif kernel == 'poly':
            line_svm = SVC(kernel=kernel, degree=poly_kernel_degree, gamma='auto')
        else:
            line_svm = linear_model.LinearRegression()

        mod_i = line_svm.fit(train_set[:, inds_in], train_set[:, inds_out])

        # Compute the training error
        train_preds_i = mod_i.predict(train_set[:, inds_in])
        train_error_i = compute_mse(train_preds_i, train_set[:, inds_out])
        train_errors.append(train_error_i)

        # Compute the testing error
        test_preds_i = mod_i.predict(test_set[:, inds_in])
        test_error_i = compute_mse(test_preds_i, test_set[:, inds_out])
        test_errors.append(test_error_i)
    return np.mean(test_errors)


def plot_svc_decision_function(SVM, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""

    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate SVM model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = SVM.decision_function(xy).reshape(X.shape)

    # Plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Plot support vectors
    if plot_support:
        ax.scatter(SVM.support_vectors_[:, 0],
                   SVM.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='k');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def get_data_bounds(data):
    """
    :param data: data to finds bounds of
    :return: x,y bounds for data
    Get bounds for data
    """
    mx_x = int(max(data[:,0]))
    mn_x = int(min(data[:,0]))
    mx_y = int(max(data[:,1]))
    mn_y = int(min(data[:,1]))
    return mn_x, mx_x, mn_y, mx_y


def random_coords(data, points):
    """
    Produce random coordinates within our data range
    :param data: data
    :param n_coordinates: how many points to return
    :return: random points within the data bound
    """
    xs = []
    ys = []
    for i in range(points):
        mn_x, mx_x, mn_y, mx_y = get_data_bounds(data)
        x = random.uniform(mn_x, mx_x)
        y = random.uniform(mn_y, mx_y)
        xs.append(x)
        ys.append(y)
    return xs, ys


def find_cluster_label(xs, ys, centers_in):
    label_xy = []

    # Cdist takes in arrays of same size
    xy = np.ones(centers_in.shape)
    xy[:,0] = xs*xy[:,0]
    xy[:,1] = ys*xy[:,1]
    xy = np.atleast_2d(xy)

    # Find distance from each point to the center of cluster to determine, in which cluster the point belongs
    distances = distance.cdist(xy, centers_in, 'euclidean')
    # Pick number of cluster with a minimal distance to the point
    for i in range(len(distances)):
        label_xy.append(np.argmin(distances[i]))
    labeled_points = np.column_stack((xy[:,0],xy[:,1], label_xy))
    return labeled_points


def test_separation():

    # Set constants
    poly_kernel_degree = 2
    k = 10  # 10 folds
    if mean_center_flag:
        data_with_classes = mean_centered(data_with_classes)

    # Indexes to split data into inputs and outputs for predictions
    inds_in = [0, 1]
    inds_out = 2

    # Dividing the data into k sets
    sets = []
    size = len(data_with_classes) // k
    for i in range(1, k + 1):
        sets.append(data_with_classes[size * (i - 1): size * i])

    # # Computing train and test errors for where
    # # each set acts as the testing data
    # Initial Train and test error lists
    train_errors = []
    test_errors = []

    # Compute for each set
    for i in range(len(sets)):
        test_set = sets[i]
        # Avoid error with concatenating arrays of shape (0,0) and (len(sets),3)
        train_set = [[0, 0, 0]]
        for j in range(len(sets)):
            if j != i:
                train_set = np.concatenate((train_set, sets[j]))
        train_set = train_set[1:]

        # Fit model to training dataset
        # Maybe there's a better way to construct to make reusable
        if kernel == 'rbf':
            line_svm = SVC(kernel=kernel, C=0.000001)
        elif kernel == 'poly':
            line_svm = SVC(kernel=kernel, degree=poly_kernel_degree, gamma='auto')
        else:
            line_svm = linear_model.LinearRegression()

        mod_i = line_svm.fit(train_set[:, inds_in], train_set[:, inds_out])

        # Compute the training error
        train_preds_i = mod_i.predict(train_set[:, inds_in])
        train_error_i = compute_mse(train_preds_i, train_set[:, inds_out])
        train_errors.append(train_error_i)

        # Compute the testing error
        test_preds_i = mod_i.predict(test_set[:, inds_in])
        test_error_i = compute_mse(test_preds_i, test_set[:, inds_out])
        test_errors.append(test_error_i)
    return np.mean(test_errors)