from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def is_edge_linking_torus(self):
    zeroes = 0
    zero_index = None
    for i in range(len(self.EdgeWeights)):
        w = self.EdgeWeights[i]
        if w == 0:
            if zeroes > 0:
                return (0, None)
            zeroes = 1
            zero_index = i
        elif w != 2:
            return (0, None)
    return (1, zero_index)