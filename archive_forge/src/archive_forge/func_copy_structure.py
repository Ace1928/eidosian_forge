from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def copy_structure(self):
    """
        Makes a copy of the structure of this BlockMatrix. This proivides a
        light-weighted copy of each block in this BlockMatrix. The blocks in the
        resulting matrix have the same shape as in the original matrices but not
        the same number of nonzeros.

        Returns
        -------
        BlockMatrix

        """
    m, n = self.bshape
    result = BlockMatrix(m, n)
    for row in range(m):
        if self.is_row_size_defined(row):
            result.set_row_size(row, self.get_row_size(row))
    for col in range(n):
        if self.is_col_size_defined(col):
            result.set_col_size(col, self.get_col_size(col))
    ii, jj = np.nonzero(self._block_mask)
    for i, j in zip(ii, jj):
        if isinstance(self._blocks[i, j], BlockMatrix):
            result.set_block(i, j, self._blocks[i, j].copy_structure())
        else:
            nrows, ncols = self._blocks[i, j].shape
            result.set_block(i, j, coo_matrix((nrows, ncols)))
    return result