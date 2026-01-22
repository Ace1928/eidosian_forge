from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def projective_distance(A, B):
    return min(matrix_norm(A - B), matrix_norm(A + B))