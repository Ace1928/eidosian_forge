import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Stochastic Block Model (SBM).

    The Stochastic Block Model graph is constructed by connecting nodes with a
    probability which depends on the cluster of the two nodes.  One can define
    the clustering association of each node, denoted by vector z, but also the
    probability matrix M.  All edge weights are equal to 1. By default, Mii >
    Mjk and nodes are uniformly clusterized.

    Parameters
    ----------
    N : int
        Number of nodes (default is 1024).
    k : float
        Number of classes (default is 5).
    z : array_like
        the vector of length N containing the association between nodes and
        classes (default is random uniform).
    M : array_like
        the k by k matrix containing the probability of connecting nodes based
        on their class belonging (default using p and q).
    p : float or array_like
        the diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a length k vector (default is p = 0.7).
    q : float or array_like
        the off-diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a k x k matrix, diagonal will be
        discarded (default is q = 0.3/k).
    directed : bool
        Allow directed edges if True (default is False).
    self_loops : bool
        Allow self loops if True (default is False).
    connected : bool
        Force the graph to be connected (default is False).
    max_iter : int
        Maximum number of trials to get a connected graph (default is 10).
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.StochasticBlockModel(
    ...     100, k=3, p=[0.4, 0.6, 0.3], q=0.02, seed=42)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.8)
    >>> G.plot(ax=axes[1])

    