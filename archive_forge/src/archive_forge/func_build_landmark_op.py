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
def build_landmark_op(self):
    """Build the landmark operator

        Calculates spectral clusters on the kernel, and calculates transition
        probabilities between cluster centers by using transition probabilities
        between samples assigned to each cluster.
        """
    with _logger.log_task('landmark operator'):
        is_sparse = sparse.issparse(self.kernel)
        with _logger.log_task('SVD'):
            _, _, VT = randomized_svd(self.diff_aff, n_components=self.n_svd, random_state=self.random_state)
        with _logger.log_task('KMeans'):
            kmeans = MiniBatchKMeans(self.n_landmark, init_size=3 * self.n_landmark, n_init=1, batch_size=10000, random_state=self.random_state)
            self._clusters = kmeans.fit_predict(self.diff_op.dot(VT.T))
        pmn = self._landmarks_to_data()
        pnm = pmn.transpose()
        pmn = normalize(pmn, norm='l1', axis=1)
        pnm = normalize(pnm, norm='l1', axis=1)
        landmark_op = pmn.dot(pnm)
        if is_sparse:
            landmark_op = landmark_op.toarray()
        self._landmark_op = landmark_op
        self._transitions = pnm