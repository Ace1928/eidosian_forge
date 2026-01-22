from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def compute_midpoint_two_horospheres_from_triangle(idealPoints, intersectionLengths):
    a, b, c = idealPoints
    la, lb = intersectionLengths
    if a == Infinity:
        return _compute_midpoint_helper(b, c, (lb / la).sqrt())
    if b == Infinity:
        return _compute_midpoint_helper(a, c, (la / lb).sqrt())
    (b, c), inv_sl_matrix = _transform_points_to_make_first_one_infinity_and_inv_sl_matrix(idealPoints)
    transformedMidpoint = _compute_midpoint_helper(b, c, (lb / la).sqrt())
    return _translate(transformedMidpoint, inv_sl_matrix)