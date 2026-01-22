from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def box_sphere_intersections(z, d, lb, ub, trust_radius, entire_line=False, extra_info=False):
    """Find the intersection between segment (or line) and box/sphere constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d``, the rectangular box
    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.
        When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.
    extra_info : bool, optional
        When ``True``, the function returns ``intersect_sphere`` and ``intersect_box``.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the rectangular box and
        inside the ball for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and both constraints. On the other hand, when ``False``, there is no
        intersection.
    sphere_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the ball. And a boolean value indicating
        whether the sphere is intersected by the line.
    box_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the box. And a boolean value indicating
        whether the box is intersected by the line.
    """
    ta_b, tb_b, intersect_b = box_intersections(z, d, lb, ub, entire_line)
    ta_s, tb_s, intersect_s = sphere_intersections(z, d, trust_radius, entire_line)
    ta = np.maximum(ta_b, ta_s)
    tb = np.minimum(tb_b, tb_s)
    if intersect_b and intersect_s and (ta <= tb):
        intersect = True
    else:
        intersect = False
    if extra_info:
        sphere_info = {'ta': ta_s, 'tb': tb_s, 'intersect': intersect_s}
        box_info = {'ta': ta_b, 'tb': tb_b, 'intersect': intersect_b}
        return (ta, tb, intersect, sphere_info, box_info)
    else:
        return (ta, tb, intersect)