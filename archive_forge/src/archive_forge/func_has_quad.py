from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def has_quad(self, tet_number):
    return max([self.get_weight(tet_number, e) for e in [E01, E02, E03]]) > 0