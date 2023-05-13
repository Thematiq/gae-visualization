import torch

import numpy as np

from sklearn.metrics import pairwise_distances


def encode_dataset(X: np.ndarray, k: int, to='cpu',
                   weights: str = 'equal', metric='euclidean'):
    """
    Transforms original dataset to a single directed graph of K nearest
    neighbours. Vertices represent samples of dataset with vertex features equal
    to coordinates in original space, and edges represent points neighbourhood

    In total for NxD dataset graph has:
    - N vertices
    - KN edges
    - D vertex features
    - 1 edge feature (optional)

    :param X: Original dataset in a form of numpy array
    :param k: Number of K nearest neighbours
    :param to: Torch device where data should be stored
    :param weights: ('equal' or 'weighted') If equal, then all edges have
        weight equal to 1, otherwise use edges weighted by distance in N
        dimensional space
    :param metric: Metric to use to calculate distance, passed to
        sklearn.metrics.pairwise_distances
    :return: 3 element tuple of:
        - Vertex features tensor
        - Edge indexes tensor
        - Edge attributes tensor
    """
    D = pairwise_distances(X, metric=metric)
    # Grab NN and flatten the matrix, so we have k-NN for 0 vertex, k-NN for 1 vertex and so on
    order = np.argsort(D)[:k, :]
    nn = order.flatten()
    edges = X.shape[0] * k

    if weights == 'equal':
        W = np.ones(shape=edges)
    else:
        W = np.take_along_axis(D, order, axis=0).flatten()

    E = np.empty(shape=(2, edges), dtype=int)
    E[1, :] = nn
    E[0, :] = np.floor_divide(np.arange(0, edges), k)
    W = torch.from_numpy(W).to(to).float()
    X = torch.from_numpy(X).to(to).float()
    E = torch.from_numpy(E).to(to).long()
    return X, E, W
