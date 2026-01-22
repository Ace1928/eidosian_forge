import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def check_branching_factor(node, branching_factor):
    subclusters = node.subclusters_
    assert branching_factor >= len(subclusters)
    for cluster in subclusters:
        if cluster.child_:
            check_branching_factor(cluster.child_, branching_factor)