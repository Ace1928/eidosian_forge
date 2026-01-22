from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def sort_clusters_by_values(clusters, values):
    """Sort `clusters` in increasing order of `values`.

    Parameters
    ----------
    clusters : array-like
        An array of cluster assignments, like the output of
        a `fit_predict()` call.
    values : type
        An associated value for each index in `clusters` to use
        for sorting the clusters.

    Returns
    -------
    new_clusters : array-likes
        Reordered cluster assignments. `np.mean(values[new_clusters == 0])`
        will be less than `np.mean(values[new_clusters == 1])` which
        will be less than `np.mean(values[new_clusters == 2])`
        and so on.

    """
    clusters = toarray(clusters)
    values = toarray(values)
    if not len(clusters) == len(values):
        raise ValueError('Expected clusters ({}) and values ({}) to be the same length.'.format(len(clusters), len(values)))
    uniq_clusters = np.unique(clusters)
    means = np.array([np.mean(values[clusters == cl]) for cl in uniq_clusters])
    new_clust_map = {curr_cl: i for i, curr_cl in enumerate(uniq_clusters[np.argsort(means)])}
    return np.array([new_clust_map[cl] for cl in clusters])