from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def block_shapes(self):
    """
        Returns list with shapes of blocks in this BlockMatrix

        Notes
        -----
        For a BlockMatrix with 2 block-rows and 2 block-cols
        this method returns [[Block_00.shape, Block_01.shape],[Block_10.shape, Block_11.shape]]

        Returns
        -------
        list

        """
    assert_block_structure(self)
    bm, bn = self.bshape
    sizes = [list() for i in range(bm)]
    for i in range(bm):
        sizes[i] = list()
        for j in range(bn):
            shape = (self._brow_lengths[i], self._bcol_lengths[j])
            sizes[i].append(shape)
    return sizes