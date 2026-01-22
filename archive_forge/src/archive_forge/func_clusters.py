from __future__ import division
from . import matrix
from . import utils
from .base import DataGraph
from .base import PyGSPGraph
from builtins import super
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numbers
import numpy as np
import tasklogger
import warnings
@property
def clusters(self):
    """Cluster assignments for each sample.

        Compute or return the cluster assignments

        Returns
        -------
        clusters : list-like, shape=[n_samples]
            Cluster assignments for each sample.
        """
    try:
        return self._clusters
    except AttributeError:
        self.build_landmark_op()
        return self._clusters