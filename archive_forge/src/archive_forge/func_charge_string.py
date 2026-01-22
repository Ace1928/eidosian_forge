from __future__ import annotations
import re
from fractions import Fraction
def charge_string(charge, brackets=True, explicit_one=True):
    """Returns a string representing the charge of an Ion. By default, the
    charge is placed in brackets with the sign preceding the magnitude, e.g.,
    '[+2]'. For uncharged species, the string returned is '(aq)'.

    Args:
        charge: the charge of the Ion
        brackets: whether to enclose the charge in brackets, e.g. [+2]. Default: True
        explicit_one: whether to include the number one for monovalent ions, e.g.
            +1 rather than +. Default: True
    """
    chg_str = '(aq)' if charge == 0 else f'{formula_double_format(charge, ignore_ones=False):+}'
    if chg_str in ['+1', '-1'] and (not explicit_one):
        chg_str = chg_str.replace('1', '')
    if chg_str != '(aq)' and brackets:
        chg_str = f'[{chg_str}]'
    return chg_str