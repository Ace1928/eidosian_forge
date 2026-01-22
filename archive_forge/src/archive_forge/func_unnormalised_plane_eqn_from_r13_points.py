from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def unnormalised_plane_eqn_from_r13_points(pts):
    """
    Given three (finite or ideal) points in the hyperboloid model
    (that is time-like or light-like vectors), compute the space-like
    vector x such that the plane defined by x * y = 0 contains the
    three given points.
    """
    return vector([_det_shifted_matrix3(pts, 0), _det_shifted_matrix3(pts, 1), -_det_shifted_matrix3(pts, 2), _det_shifted_matrix3(pts, 3)])