from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def transform_to_R3(self, matrix, bdry_map=None):
    new_start = self.start.transform_to_R3(matrix, bdry_map)
    new_end = self.end.transform_to_R3(matrix, bdry_map)
    return BarycentricArc(new_start, new_end)