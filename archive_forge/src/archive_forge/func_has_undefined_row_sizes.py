from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def has_undefined_row_sizes(self):
    """
        Indicates if the matrix has block-rows with undefined dimensions

        Returns
        -------
        bool

        """
    return len(self._undefined_brows) != 0