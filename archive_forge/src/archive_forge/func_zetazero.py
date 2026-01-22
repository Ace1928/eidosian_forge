from .functions import defun, defun_wrapped
@defun
def zetazero(ctx, n, info=False, round=True):
    """
    Computes the `n`-th nontrivial zero of `\\zeta(s)` on the critical line,
    i.e. returns an approximation of the `n`-th largest complex number
    `s = \\frac{1}{2} + ti` for which `\\zeta(s) = 0`. Equivalently, the
    imaginary part `t` is a zero of the Z-function (:func:`~mpmath.siegelz`).

    **Examples**

    The first few zeros::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> zetazero(1)
        (0.5 + 14.13472514173469379045725j)
        >>> zetazero(2)
        (0.5 + 21.02203963877155499262848j)
        >>> zetazero(20)
        (0.5 + 77.14484006887480537268266j)

    Verifying that the values are zeros::

        >>> for n in range(1,5):
        ...     s = zetazero(n)
        ...     chop(zeta(s)), chop(siegelz(s.imag))
        ...
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)

    Negative indices give the conjugate zeros (`n = 0` is undefined)::

        >>> zetazero(-1)
        (0.5 - 14.13472514173469379045725j)

    :func:`~mpmath.zetazero` supports arbitrarily large `n` and arbitrary precision::

        >>> mp.dps = 15
        >>> zetazero(1234567)
        (0.5 + 727690.906948208j)
        >>> mp.dps = 50
        >>> zetazero(1234567)
        (0.5 + 727690.9069482075392389420041147142092708393819935j)
        >>> chop(zeta(_)/_)
        0.0

    with *info=True*, :func:`~mpmath.zetazero` gives additional information::

        >>> mp.dps = 15
        >>> zetazero(542964976,info=True)
        ((0.5 + 209039046.578535j), [542964969, 542964978], 6, '(013111110)')

    This means that the zero is between Gram points 542964969 and 542964978;
    it is the 6-th zero between them. Finally (01311110) is the pattern
    of zeros in this interval. The numbers indicate the number of zeros
    in each Gram interval (Rosser blocks between parenthesis). In this case
    there is only one Rosser block of length nine.
    """
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError('n must be nonzero')
    wpinitial = ctx.prec
    try:
        wpz, fp_tolerance = comp_fp_tolerance(ctx, n)
        ctx.prec = wpz
        if n < 400000000:
            my_zero_number, block, T, V = find_rosser_block_zero(ctx, n)
        else:
            my_zero_number, block, T, V = search_supergood_block(ctx, n, fp_tolerance)
        zero_number_block = block[1] - block[0]
        T, V, separated = separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=ctx.inf, fp_tolerance=fp_tolerance)
        if info:
            pattern = pattern_construct(ctx, block, T, V)
        prec = max(wpinitial, wpz)
        t = separate_my_zero(ctx, my_zero_number, zero_number_block, T, V, prec)
        v = ctx.mpc(0.5, t)
    finally:
        ctx.prec = wpinitial
    if round:
        v = +v
    if info:
        return (v, block, my_zero_number, pattern)
    else:
        return v