from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def to_3d_points(self):
    return (self.start.to_3d_point(), self.end.to_3d_point())