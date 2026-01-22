from .functions import defun, defun_wrapped
@defun_wrapped
def qfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    """
    Returns the elliptic nome `q`, given any of `q, m, k, \\tau, \\bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfrom(q=0.25)
        0.25
        >>> qfrom(m=mfrom(q=0.25))
        0.25
        >>> qfrom(k=kfrom(q=0.25))
        0.25
        >>> qfrom(tau=taufrom(q=0.25))
        (0.25 + 0.0j)
        >>> qfrom(qbar=qbarfrom(q=0.25))
        0.25

    """
    if q is not None:
        return ctx.convert(q)
    if m is not None:
        return nome(ctx, m)
    if k is not None:
        return nome(ctx, ctx.convert(k) ** 2)
    if tau is not None:
        return ctx.expjpi(tau)
    if qbar is not None:
        return ctx.sqrt(qbar)