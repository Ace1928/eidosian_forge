from __future__ import annotations
import re
from fractions import Fraction
def formula_double_format(afloat, ignore_ones=True, tol: float=1e-08):
    """This function is used to make pretty formulas by formatting the amounts.
    Instead of Li1.0 Fe1.0 P1.0 O4.0, you get LiFePO4.

    Args:
        afloat (float): a float
        ignore_ones (bool): if true, floats of 1 are ignored.
        tol (float): Tolerance to round to nearest int. i.e. 2.0000000001 -> 2

    Returns:
        A string representation of the float for formulas.
    """
    if ignore_ones and afloat == 1:
        return ''
    if abs(afloat - int(afloat)) < tol:
        return int(afloat)
    return round(afloat, 8)