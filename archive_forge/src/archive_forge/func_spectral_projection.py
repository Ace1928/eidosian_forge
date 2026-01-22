from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def spectral_projection(u, eigenpairs):
    """
    Returns the coefficients of each eigenvector
    in a projection of the vector u onto the normalized
    eigenvectors which are contained in eigenpairs.

    eigenpairs should be a list of two objects.  The
    first is a list of eigenvalues and the second a list
    of eigenvectors.  The eigenvectors should be lists.

    There's not a lot of error checking on lengths of
    arrays, etc. so be careful.
    """
    coeff = []
    evect = eigenpairs[1]
    for ev in evect:
        c = sum((evv * uv for evv, uv in zip(ev, u)))
        coeff.append(c)
    return coeff