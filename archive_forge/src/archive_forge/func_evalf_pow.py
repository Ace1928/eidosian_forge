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
def evalf_pow(v: 'Pow', prec: int, options) -> TMP_RES:
    target_prec = prec
    base, exp = v.args
    if exp.is_Integer:
        p: int = exp.p
        if not p:
            return (fone, None, prec, None)
        prec += int(math.log(abs(p), 2))
        result = evalf(base, prec + 5, options)
        if result is S.ComplexInfinity:
            if p < 0:
                return (None, None, None, None)
            return result
        re, im, re_acc, im_acc = result
        if re and (not im):
            return (mpf_pow_int(re, p, target_prec), None, target_prec, None)
        if im and (not re):
            z = mpf_pow_int(im, p, target_prec)
            case = p % 4
            if case == 0:
                return (z, None, target_prec, None)
            if case == 1:
                return (None, z, None, target_prec)
            if case == 2:
                return (mpf_neg(z), None, target_prec, None)
            if case == 3:
                return (None, mpf_neg(z), None, target_prec)
        if not re:
            if p < 0:
                return S.ComplexInfinity
            return (None, None, None, None)
        re, im = libmp.mpc_pow_int((re, im), p, prec)
        return finalize_complex(re, im, target_prec)
    result = evalf(base, prec + 5, options)
    if result is S.ComplexInfinity:
        if exp.is_Rational:
            if exp < 0:
                return (None, None, None, None)
            return result
        raise NotImplementedError
    if exp is S.Half:
        xre, xim, _, _ = result
        if xim:
            re, im = libmp.mpc_sqrt((xre or fzero, xim), prec)
            return finalize_complex(re, im, prec)
        if not xre:
            return (None, None, None, None)
        if mpf_lt(xre, fzero):
            return (None, mpf_sqrt(mpf_neg(xre), prec), None, prec)
        return (mpf_sqrt(xre, prec), None, prec, None)
    prec += 10
    result = evalf(exp, prec, options)
    if result is S.ComplexInfinity:
        return (fnan, None, prec, None)
    yre, yim, _, _ = result
    if not (yre or yim):
        return (fone, None, prec, None)
    ysize = fastlog(yre)
    if ysize > 5:
        prec += ysize
        yre, yim, _, _ = evalf(exp, prec, options)
    if base is S.Exp1:
        if yim:
            re, im = libmp.mpc_exp((yre or fzero, yim), prec)
            return finalize_complex(re, im, target_prec)
        return (mpf_exp(yre, target_prec), None, target_prec, None)
    xre, xim, _, _ = evalf(base, prec + 5, options)
    if not (xre or xim):
        if yim:
            return (fnan, None, prec, None)
        if yre[0] == 1:
            return S.ComplexInfinity
        return (None, None, None, None)
    if yim:
        re, im = libmp.mpc_pow((xre or fzero, xim or fzero), (yre or fzero, yim), target_prec)
        return finalize_complex(re, im, target_prec)
    if xim:
        re, im = libmp.mpc_pow_mpf((xre or fzero, xim), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    elif mpf_lt(xre, fzero):
        re, im = libmp.mpc_pow_mpf((xre, fzero), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    else:
        return (mpf_pow(xre, yre, target_prec), None, target_prec, None)