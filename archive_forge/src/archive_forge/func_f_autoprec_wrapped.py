import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def f_autoprec_wrapped(*args, **kwargs):
    prec = ctx.prec
    if maxprec is None:
        maxprec2 = ctx._default_hyper_maxprec(prec)
    else:
        maxprec2 = maxprec
    try:
        ctx.prec = prec + 10
        try:
            v1 = f(*args, **kwargs)
        except catch:
            v1 = ctx.nan
        prec2 = prec + 20
        while 1:
            ctx.prec = prec2
            try:
                v2 = f(*args, **kwargs)
            except catch:
                v2 = ctx.nan
            if v1 == v2:
                break
            err = ctx.mag(v2 - v1) - ctx.mag(v2)
            if err < -prec:
                break
            if verbose:
                print('autoprec: target=%s, prec=%s, accuracy=%s' % (prec, prec2, -err))
            v1 = v2
            if prec2 >= maxprec2:
                raise ctx.NoConvergence('autoprec: prec increased to %i without convergence' % prec2)
            prec2 += int(prec2 * 2)
            prec2 = min(prec2, maxprec2)
    finally:
        ctx.prec = prec
    return +v2