from .simplex import *
from .perm4 import Perm4, inv
def glued(self):
    a = self.copy()
    if a.next() is None:
        a.Tetrahedron = None
    return a