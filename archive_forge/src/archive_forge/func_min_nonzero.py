from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def min_nonzero(self):
    return min([c for c in self.vector if c > 0])