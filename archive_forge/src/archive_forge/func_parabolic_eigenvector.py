from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def parabolic_eigenvector(A):
    P, CC, epsilon = (make_trace_2(A), A.base_ring(), make_epsilon(A))
    for v in [vector(CC, (1, 0)), vector(CC, (0, 1))]:
        if (v - P * v).norm() < epsilon:
            return v
    v = vector(CC, pari(P - 1).matker()[0])
    assert (v - P * v).norm() < epsilon
    return v