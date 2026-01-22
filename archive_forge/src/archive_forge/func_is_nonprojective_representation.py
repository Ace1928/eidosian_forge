from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def is_nonprojective_representation(self):
    """
        True if this is an SL(2,C)-representation, i.e., if multiplying the generators
        in a word yields the identity matrix.
        """
    return all((is_essentially_Id2(self(R)) for R in self.relators()))