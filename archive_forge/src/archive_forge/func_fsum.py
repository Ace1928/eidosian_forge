from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def fsum(ctx, terms, absolute=False, squared=False):
    """
        Calculates a sum containing a finite number of terms (for infinite
        series, see :func:`~mpmath.nsum`). The terms will be converted to
        mpmath numbers. For len(terms) > 2, this function is generally
        faster and produces more accurate results than the builtin
        Python function :func:`sum`.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsum([1, 2, 0.5, 7])
            mpf('10.5')

        With squared=True each term is squared, and with absolute=True
        the absolute value of each term is used.
        """
    prec, rnd = ctx._prec_rounding
    real = []
    imag = []
    for term in terms:
        reval = imval = 0
        if hasattr(term, '_mpf_'):
            reval = term._mpf_
        elif hasattr(term, '_mpc_'):
            reval, imval = term._mpc_
        else:
            term = ctx.convert(term)
            if hasattr(term, '_mpf_'):
                reval = term._mpf_
            elif hasattr(term, '_mpc_'):
                reval, imval = term._mpc_
            else:
                raise NotImplementedError
        if imval:
            if squared:
                if absolute:
                    real.append(mpf_mul(reval, reval))
                    real.append(mpf_mul(imval, imval))
                else:
                    reval, imval = mpc_pow_int((reval, imval), 2, prec + 10)
                    real.append(reval)
                    imag.append(imval)
            elif absolute:
                real.append(mpc_abs((reval, imval), prec))
            else:
                real.append(reval)
                imag.append(imval)
        else:
            if squared:
                reval = mpf_mul(reval, reval)
            elif absolute:
                reval = mpf_abs(reval)
            real.append(reval)
    s = mpf_sum(real, prec, rnd, absolute)
    if imag:
        s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
    else:
        s = ctx.make_mpf(s)
    return s