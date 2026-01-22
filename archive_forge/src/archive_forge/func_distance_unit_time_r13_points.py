from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def distance_unit_time_r13_points(u, v):
    """
    Computes the hyperbolic distance between two points (represented
    by unit time vectors) in the hyperboloid model.
    """
    d = -r13_dot(u, v)
    if is_RealIntervalFieldElement(d):
        RIF = d.parent()
        d = d.intersection(RIF(1, sage.all.Infinity))
    elif d < 1:
        RF = d.parent()
        d = RF(1)
    return d.arccosh()