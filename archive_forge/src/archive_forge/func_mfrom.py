from .functions import defun, defun_wrapped
@defun_wrapped
def mfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    """
    Returns the elliptic parameter `m`, given any of
    `q, m, k, \\tau, \\bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> mfrom(m=0.25)
        0.25
        >>> mfrom(q=qfrom(m=0.25))
        0.25
        >>> mfrom(k=kfrom(m=0.25))
        0.25
        >>> mfrom(tau=taufrom(m=0.25))
        (0.25 + 0.0j)
        >>> mfrom(qbar=qbarfrom(m=0.25))
        0.25

    As `q \\to 1` and `q \\to -1`, `m` rapidly approaches
    `1` and `-\\infty` respectively::

        >>> mfrom(q=0.75)
        0.9999999999999798332943533
        >>> mfrom(q=-0.75)
        -49586681013729.32611558353
        >>> mfrom(q=1)
        1.0
        >>> mfrom(q=-1)
        -inf

    The inverse nome as a function of `q` has an integer
    Taylor series expansion::

        >>> taylor(lambda q: mfrom(q), 0, 7)
        [0.0, 16.0, -128.0, 704.0, -3072.0, 11488.0, -38400.0, 117632.0]

    """
    if m is not None:
        return m
    if k is not None:
        return k ** 2
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return ctx.convert(q)
    if q == -1:
        return q * ctx.inf
    v = (ctx.jtheta(2, 0, q) / ctx.jtheta(3, 0, q)) ** 4
    if ctx._is_real_type(q) and q < 0:
        v = v.real
    return v