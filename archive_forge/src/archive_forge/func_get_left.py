import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def get_left(self):
    """
        Return a reference to the left child tree object.

        Returns
        -------
        left : ClusterNode
            The left child of the target node. If the node is a leaf,
            None is returned.

        """
    return self.left