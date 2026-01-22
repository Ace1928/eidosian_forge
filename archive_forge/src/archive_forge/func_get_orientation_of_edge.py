from .simplex import *
from .perm4 import Perm4, inv
import sys
def get_orientation_of_edge(self, a, b):
    return self.Class[a | b].orientation_with_respect_to(self, a, b)