from .simplex import *
from .perm4 import Perm4, inv
def south_tail(self):
    return self.Tetrahedron.Class[self.tail() | OppTail[self.tail(), self.head()]]