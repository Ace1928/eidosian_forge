from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
import math
import mpmath.libmp as libmp
from mpmath import (
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int
def chop_parts(value: TMP_RES, prec: int) -> TMP_RES:
    """
    Chop off tiny real or complex parts.
    """
    if value is S.ComplexInfinity:
        return value
    re, im, re_acc, im_acc = value
    if re and re not in _infs_nan and (fastlog(re) < -prec + 4):
        re, re_acc = (None, None)
    if im and im not in _infs_nan and (fastlog(im) < -prec + 4):
        im, im_acc = (None, None)
    if re and im:
        delta = fastlog(re) - fastlog(im)
        if re_acc < 2 and delta - re_acc <= -prec + 4:
            re, re_acc = (None, None)
        if im_acc < 2 and delta - im_acc >= prec - 4:
            im, im_acc = (None, None)
    return (re, im, re_acc, im_acc)