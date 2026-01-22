from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def boundary_face(self):
    zeros = self._zero_coordinates
    if len(zeros) != 1:
        raise GeneralPositionError('Not a generic point on a face')
    return zeros[0]