from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def compute_midpoint_of_triangle_edge_with_offset(idealPoints, offset):
    """
    The inputs are a list of three IdealPoint's [a, b, c] and an element
    offset in RealIntervalField.

    Consider the triangle spanned by the three ideal points. There is a line
    from c perpendicular to the side a b. Call the intersection of the line
    with the side a b the midpoint. This function returns this point moved
    towards a by hyperbolic distance log(offset)::

        sage: from sage.all import CIF, RIF
        sage: compute_midpoint_of_triangle_edge_with_offset( # doctest: +NUMERIC12
        ...       [ CIF(0), Infinity, CIF(1) ], RIF(5.0))
        FinitePoint(0, 0.2000000000000000?)

    """
    a, b, c = idealPoints
    if a == Infinity:
        return _compute_midpoint_helper(b, c, offset)
    if b == Infinity:
        return _compute_midpoint_helper(a, c, 1 / offset)
    (b, c), inv_sl_matrix = _transform_points_to_make_first_one_infinity_and_inv_sl_matrix(idealPoints)
    transformedMidpoint = _compute_midpoint_helper(b, c, offset)
    return _translate(transformedMidpoint, inv_sl_matrix)