from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def fdot(ctx, A, B=None, conjugate=False):
    """
        Computes the dot product of the iterables `A` and `B`,

        .. math ::

            \\sum_{k=0} A_k B_k.

        Alternatively, :func:`~mpmath.fdot` accepts a single iterable of pairs.
        In other words, ``fdot(A,B)`` and ``fdot(zip(A,B))`` are equivalent.
        The elements are automatically converted to mpmath numbers.

        With ``conjugate=True``, the elements in the second vector
        will be conjugated:

        .. math ::

            \\sum_{k=0} A_k \\overline{B_k}

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = [2, 1.5, 3]
            >>> B = [1, -1, 2]
            >>> fdot(A, B)
            mpf('6.5')
            >>> list(zip(A, B))
            [(2, 1), (1.5, -1), (3, 2)]
            >>> fdot(_)
            mpf('6.5')
            >>> A = [2, 1.5, 3j]
            >>> B = [1+j, 3, -1-j]
            >>> fdot(A, B)
            mpc(real='9.5', imag='-1.0')
            >>> fdot(A, B, conjugate=True)
            mpc(real='3.5', imag='-5.0')

        """
    if B is not None:
        A = zip(A, B)
    prec, rnd = ctx._prec_rounding
    real = []
    imag = []
    hasattr_ = hasattr
    types = (ctx.mpf, ctx.mpc)
    for a, b in A:
        if type(a) not in types:
            a = ctx.convert(a)
        if type(b) not in types:
            b = ctx.convert(b)
        a_real = hasattr_(a, '_mpf_')
        b_real = hasattr_(b, '_mpf_')
        if a_real and b_real:
            real.append(mpf_mul(a._mpf_, b._mpf_))
            continue
        a_complex = hasattr_(a, '_mpc_')
        b_complex = hasattr_(b, '_mpc_')
        if a_real and b_complex:
            aval = a._mpf_
            bre, bim = b._mpc_
            if conjugate:
                bim = mpf_neg(bim)
            real.append(mpf_mul(aval, bre))
            imag.append(mpf_mul(aval, bim))
        elif b_real and a_complex:
            are, aim = a._mpc_
            bval = b._mpf_
            real.append(mpf_mul(are, bval))
            imag.append(mpf_mul(aim, bval))
        elif a_complex and b_complex:
            are, aim = a._mpc_
            bre, bim = b._mpc_
            if conjugate:
                bim = mpf_neg(bim)
            real.append(mpf_mul(are, bre))
            real.append(mpf_neg(mpf_mul(aim, bim)))
            imag.append(mpf_mul(are, bim))
            imag.append(mpf_mul(aim, bre))
        else:
            raise NotImplementedError
    s = mpf_sum(real, prec, rnd)
    if imag:
        s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
    else:
        s = ctx.make_mpf(s)
    return s