import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Interval
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted
from . import AgglomerativeClustering
def merge_subcluster(self, nominee_cluster, threshold):
    """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
    new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
    new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
    new_n = self.n_samples_ + nominee_cluster.n_samples_
    new_centroid = 1 / new_n * new_ls
    new_sq_norm = np.dot(new_centroid, new_centroid)
    sq_radius = new_ss / new_n - new_sq_norm
    if sq_radius <= threshold ** 2:
        self.n_samples_, self.linear_sum_, self.squared_sum_, self.centroid_, self.sq_norm_ = (new_n, new_ls, new_ss, new_centroid, new_sq_norm)
        return True
    return False