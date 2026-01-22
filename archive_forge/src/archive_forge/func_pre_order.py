import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def pre_order(self, func=lambda x: x.id):
    """
        Perform pre-order traversal without recursive function calls.

        When a leaf node is first encountered, ``func`` is called with
        the leaf node as its argument, and its result is appended to
        the list.

        For example, the statement::

           ids = root.pre_order(lambda x: x.id)

        returns a list of the node ids corresponding to the leaf nodes
        of the tree as they appear from left to right.

        Parameters
        ----------
        func : function
            Applied to each leaf ClusterNode object in the pre-order traversal.
            Given the ``i``-th leaf node in the pre-order traversal ``n[i]``,
            the result of ``func(n[i])`` is stored in ``L[i]``. If not
            provided, the index of the original observation to which the node
            corresponds is used.

        Returns
        -------
        L : list
            The pre-order traversal.

        """
    n = self.count
    curNode = [None] * (2 * n)
    lvisited = set()
    rvisited = set()
    curNode[0] = self
    k = 0
    preorder = []
    while k >= 0:
        nd = curNode[k]
        ndid = nd.id
        if nd.is_leaf():
            preorder.append(func(nd))
            k = k - 1
        elif ndid not in lvisited:
            curNode[k + 1] = nd.left
            lvisited.add(ndid)
            k = k + 1
        elif ndid not in rvisited:
            curNode[k + 1] = nd.right
            rvisited.add(ndid)
            k = k + 1
        else:
            k = k - 1
    return preorder