import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans


def normalize(data):
    """
    :param data: input data needed to normalize. Important: input only columns that need to be normalized
    :return: normalized data
    """
    for i in range(len(data[0])):
        mx = np.max(data[:, i])
        mn = np.min(data[:, i])
        data[:, i] = (data[:, i] - mn) / (mx - mn)
        data[:, i] = np.around(data[:, i].astype(np.double), decimals=2)
    return data


def get_clusters_tightness_for_ks(data, list):
    """
    :param data: Input data
    :param list: Potential number of clusters
    :return: list of total spread measures for all potential k's
    """
    data = normalize(data)
    total_spread_per_k = []
    # Loop over a list of potential numbers of clusters to do kmeans on each
    for i in list:
        km_alg = KMeans(n_clusters=i, init="random", random_state=1, max_iter=200)
        fit1 = km_alg.fit(data)
        labels = fit1.labels_
        centers = fit1.cluster_centers_
        inds = [*range(i)]
        spread_per_k_selection = []

        # For each cluster compute total cluster spread and sum it up to get total spread
        for j in range(i):
            # Marks all data contained in a cluster
            inds[j] = labels == j
            # Defines cluster by the data in it
            cluster = data[inds[j]].astype(np.float64)

            center = np.atleast_2d([centers[j, :]])
            cluster_data_distances = distance.cdist(cluster, center, 'euclidean')
            # adds total spread on cluster
            cluster_spread = np.sum(cluster_data_distances)
            spread_per_k_selection.append(cluster_spread)
        total_spread_per_k.append(sum(spread_per_k_selection))
    return total_spread_per_k
