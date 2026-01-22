from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def cross_ratio(z0, z1, z2, z3):
    """
    Computes the cross ratio (according to SnapPea conventions) of
    four ideal points::

        sage: from sage.all import CIF
        sage: cross_ratio(Infinity, CIF(0), CIF(1), CIF(1.2, 1.3)) # doctest: +NUMERIC12
        1.2000000000000000? + 1.300000000000000?*I

    """
    return _diff_1_if_inf(z2, z0) * _diff_1_if_inf(z3, z1) / (_diff_1_if_inf(z2, z1) * _diff_1_if_inf(z3, z0))