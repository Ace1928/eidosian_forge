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
def evalf_add(v: 'Add', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)
        im, _, im_acc, _ = evalf(c, prec, options)
        return (re, im, re_acc, im_acc)
    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2 * prec)
        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        n = terms.count(S.ComplexInfinity)
        if n >= 2:
            return (fnan, None, prec, None)
        re, re_acc = add_terms([a[0::2] for a in terms if isinstance(a, tuple) and a[0]], prec, target_prec)
        im, im_acc = add_terms([a[1::2] for a in terms if isinstance(a, tuple) and a[1]], prec, target_prec)
        if n == 1:
            if re in (finf, fninf, fnan) or im in (finf, fninf, fnan):
                return (fnan, None, prec, None)
            return S.ComplexInfinity
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print('ADD: wanted', target_prec, 'accurate bits, got', re_acc, im_acc)
            break
        else:
            if prec - target_prec > options['maxprec']:
                break
            prec = prec + max(10 + 2 ** i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print('ADD: restarting with prec', prec)
    options['maxprec'] = oldmaxprec
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return (re, im, re_acc, im_acc)