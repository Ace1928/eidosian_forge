import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def max_distance_point(self, x, p=2.0):
    """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
    return minkowski_distance(0, np.maximum(self.maxes - x, x - self.mins), p)