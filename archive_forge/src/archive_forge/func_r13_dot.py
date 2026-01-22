from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def r13_dot(u, v):
    """
    -+++ inner product of two 4-vectors.
    """
    return -u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3]