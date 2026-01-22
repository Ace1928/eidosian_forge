from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def assert_block_structure(mat):
    if mat.has_undefined_row_sizes():
        msgr = 'Operation not allowed with None rows. Specify at least one block in every row'
        raise NotFullyDefinedBlockMatrixError(msgr)
    if mat.has_undefined_col_sizes():
        msgc = 'Operation not allowed with None columns. Specify at least one block every column'
        raise NotFullyDefinedBlockMatrixError(msgc)