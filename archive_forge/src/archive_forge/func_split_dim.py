import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
@property
def split_dim(self):
    return self._node.split_dim