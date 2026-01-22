import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def strand_matrix_merge(A, a, b):
    """
    The main computations all happen in this method.  Here A is a
    square matrix and a and b are row (equivalently column) indices.
    """
    assert a != b
    alpha, beta = (A[a, a], A[a, b])
    gamma, delta = (A[b, a], A[b, b])
    mu = 1 - beta
    theta, epsilon = (A.row(a), A.row(b))
    phi, psi = (A.column(a), A.column(b))
    A = A + matrix(psi).transpose() * matrix(theta) / mu
    i, j = (min(a, b), max(a, b))
    A[i] = epsilon + delta * theta / mu
    A[:, i] = phi + alpha * psi / mu
    A[i, i] = gamma + alpha * delta / mu
    A = A.delete_rows([j])
    A = A.delete_columns([j])
    return A