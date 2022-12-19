from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance


def full_kmeans(data, k):
    """
    :param data: data to cluster
    :param k: number of cluster to separate data into
    :return: labels indentifying points belonging to a cluster, cluster centers
    """
    km_alg = KMeans(n_clusters=k, init="random", random_state=70, max_iter=200)
    fit = km_alg.fit(data)
    return fit.labels_, fit.cluster_centers_


def make_adj(data):
    """
    :param data: observations that we are separating
    :return: adjacency matrix
    """
    # Compute the pairwise distances between each pair of data points
    # Any pair of points that are within a certain distance of each other
    # We will consider "close", and any pair of points that are not within that distance are "far".
    distances = distance.cdist(data, data, 'euclidean')
    inds = distances < 1 / 2

    # If distance is less than 1/2, the datapoint is labeled 1, otherwise 0
    n_rows = distances.shape[0]
    adj_m = np.zeros([n_rows, n_rows])
    adj_m[inds] = 1

    # Set the matrix diagonal labels to 0
    diag = np.diag(np.diag(adj_m))
    adj_m = adj_m - diag
    return adj_m


def my_laplacian(adj_m):
    """
    :param adj_m: Adjacency matrix, showing which points each point is nearing
    :return: Laplacian matrix representing how many points given point is nearing
    """
    # Counts the number of datapoints that each data point is near.
    colsum = np.sum(adj_m, axis=0)

    # Makes a diagonal degree matrix that specifies how many points a given point is neighbouring with
    degree_diag = np.diag(colsum)
    # Computes the unnormalized Laplacian:
    # L = D - A
    L = degree_diag - adj_m
    return L


def spect_clustering(L, k):
    # Computing the eigenvectors of L
    eigvals, eigvecs = np.linalg.eig(L)
    # Order the eigenvalues from smallest to greatest, and place the eigenvectors in the same order
    inds = np.abs(eigvals).argsort()
    ordered_eigvecs = eigvecs[:, inds]
    # Select the first k eigenvectors
    # Each row of the resulting rectangular matrix is a new k-dimensional representation/summary for each datapoint.
    firstK = ordered_eigvecs[:, :k]
    # Compute k-means on the new k dimensional representation
    labels, centers = full_kmeans(firstK, k)
    return labels, centers