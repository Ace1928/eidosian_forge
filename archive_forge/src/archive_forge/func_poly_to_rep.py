from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def poly_to_rep(f):
    """
    For a polynomial f in F[x], return the matrix corresponding to the
    left action of x on F[x]/(f).
    """
    assert f.is_monic()
    x = f.parent().gen()
    d = f.degree()
    last_column = [-f[e] for e in range(d)]
    I = identity_matrix(d)
    M = matrix(I.rows()[1:] + [last_column])
    assert M.charpoly() == f
    return M.transpose()