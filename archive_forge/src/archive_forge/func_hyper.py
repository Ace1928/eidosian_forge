from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyper(ctx, a_s, b_s, z, **kwargs):
    """
    Hypergeometric function, general case.
    """
    z = ctx.convert(z)
    p = len(a_s)
    q = len(b_s)
    a_s = [ctx._convert_param(a) for a in a_s]
    b_s = [ctx._convert_param(b) for b in b_s]
    if kwargs.get('eliminate', True):
        elim_nonpositive = kwargs.get('eliminate_all', False)
        i = 0
        while i < q and a_s:
            b = b_s[i]
            if b in a_s and (elim_nonpositive or not ctx.isnpint(b[0])):
                a_s.remove(b)
                b_s.remove(b)
                p -= 1
                q -= 1
            else:
                i += 1
    if p == 0:
        if q == 1:
            return ctx._hyp0f1(b_s, z, **kwargs)
        elif q == 0:
            return ctx.exp(z)
    elif p == 1:
        if q == 1:
            return ctx._hyp1f1(a_s, b_s, z, **kwargs)
        elif q == 2:
            return ctx._hyp1f2(a_s, b_s, z, **kwargs)
        elif q == 0:
            return ctx._hyp1f0(a_s[0][0], z)
    elif p == 2:
        if q == 1:
            return ctx._hyp2f1(a_s, b_s, z, **kwargs)
        elif q == 2:
            return ctx._hyp2f2(a_s, b_s, z, **kwargs)
        elif q == 3:
            return ctx._hyp2f3(a_s, b_s, z, **kwargs)
        elif q == 0:
            return ctx._hyp2f0(a_s, b_s, z, **kwargs)
    elif p == q + 1:
        return ctx._hypq1fq(p, q, a_s, b_s, z, **kwargs)
    elif p > q + 1 and (not kwargs.get('force_series')):
        return ctx._hyp_borel(p, q, a_s, b_s, z, **kwargs)
    coeffs, types = zip(*a_s + b_s)
    return ctx.hypsum(p, q, types, coeffs, z, **kwargs)