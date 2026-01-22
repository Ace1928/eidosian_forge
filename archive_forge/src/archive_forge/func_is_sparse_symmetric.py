import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def is_sparse_symmetric(m, complex: bool=False) -> bool:
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')
    if not isinstance(m, sp.coo_matrix):
        m = sp.coo_matrix(m)
    r, c, v = (m.row, m.col, m.data)
    tril_no_diag = r > c
    triu_no_diag = c > r
    if triu_no_diag.sum() != tril_no_diag.sum():
        return False
    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]
    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]
    if complex:
        check = np.allclose(vl, np.conj(vu))
    else:
        check = np.allclose(vl, vu)
    return check