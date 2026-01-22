from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def set_row_size(self, row, size):
    if row in self._undefined_brows:
        self._undefined_brows.remove(row)
        self._brow_lengths[row] = size
        if len(self._undefined_brows) == 0:
            self._brow_lengths = np.asarray(self._brow_lengths, dtype=np.int64)
    elif self._brow_lengths[row] != size:
        raise ValueError('Incompatible row dimensions for row {row}; got {got}; expected {exp}'.format(row=row, got=size, exp=self._brow_lengths[row]))