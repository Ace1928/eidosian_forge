import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
def make_lower_triangular_full(lower_triangular_matrix):
    """
    This function takes a symmetric matrix that only has entries in the
    lower triangle and makes is a full matrix by duplicating the entries
    """
    mask = lower_triangular_matrix.row != lower_triangular_matrix.col
    row = np.concatenate((lower_triangular_matrix.row, lower_triangular_matrix.col[mask]))
    col = np.concatenate((lower_triangular_matrix.col, lower_triangular_matrix.row[mask]))
    data = np.concatenate((lower_triangular_matrix.data, lower_triangular_matrix.data[mask]))
    return coo_matrix((data, (row, col)), shape=lower_triangular_matrix.shape)