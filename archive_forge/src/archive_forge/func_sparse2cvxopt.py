import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def sparse2cvxopt(value):
    """Converts a SciPy sparse matrix to a CVXOPT sparse matrix.

    Parameters
    ----------
    sparse_mat : SciPy sparse matrix
        The matrix to convert.

    Returns
    -------
    CVXOPT spmatrix
        The converted matrix.
    """
    import cvxopt
    if isinstance(value, (np.ndarray, np.matrix)):
        return cvxopt.sparse(cvxopt.matrix(value.astype('float64')), tc='d')
    elif sp.issparse(value):
        value = value.tocoo()
        return cvxopt.spmatrix(value.data.tolist(), value.row.tolist(), value.col.tolist(), size=value.shape, tc='d')