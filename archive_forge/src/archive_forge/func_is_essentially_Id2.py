from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def is_essentially_Id2(M, error=10 ** (-3)):
    diff = M - identity(M)
    error = diff.base_ring()(error)
    return all((abs(d) < error for d in diff.list()))