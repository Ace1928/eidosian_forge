from .functions import defun, defun_wrapped
@defun_wrapped
def kfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    """
    Returns the elliptic modulus `k`, given any of
    `q, m, k, \\tau, \\bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> kfrom(k=0.25)
        0.25
        >>> kfrom(m=mfrom(k=0.25))
        0.25
        >>> kfrom(q=qfrom(k=0.25))
        0.25
        >>> kfrom(tau=taufrom(k=0.25))
        (0.25 + 0.0j)
        >>> kfrom(qbar=qbarfrom(k=0.25))
        0.25

    As `q \\to 1` and `q \\to -1`, `k` rapidly approaches
    `1` and `i \\infty` respectively::

        >>> kfrom(q=0.75)
        0.9999999999999899166471767
        >>> kfrom(q=-0.75)
        (0.0 + 7041781.096692038332790615j)
        >>> kfrom(q=1)
        1
        >>> kfrom(q=-1)
        (0.0 + +infj)
    """
    if k is not None:
        return ctx.convert(k)
    if m is not None:
        return ctx.sqrt(m)
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return q
    if q == -1:
        return ctx.mpc(0, 'inf')
    return (ctx.jtheta(2, 0, q) / ctx.jtheta(3, 0, q)) ** 2