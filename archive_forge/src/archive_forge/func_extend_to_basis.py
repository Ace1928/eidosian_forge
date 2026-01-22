from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def extend_to_basis(v):
    u = 1 / v.norm() * v
    w = vector(v.base_ring(), (-u[1].conjugate(), u[0].conjugate()))
    return matrix([u, w]).transpose()