from .functions import defun, defun_wrapped
@defun
def nzeros(ctx, t):
    """
    Computes the number of zeros of the Riemann zeta function in
    `(0,1) \\times (0,t]`, usually denoted by `N(t)`.

    **Examples**

    The first zero has imaginary part between 14 and 15::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nzeros(14)
        0
        >>> nzeros(15)
        1
        >>> zetazero(1)
        (0.5 + 14.1347251417347j)

    Some closely spaced zeros::

        >>> nzeros(10**7)
        21136125
        >>> zetazero(21136125)
        (0.5 + 9999999.32718175j)
        >>> zetazero(21136126)
        (0.5 + 10000000.2400236j)
        >>> nzeros(545439823.215)
        1500000001
        >>> zetazero(1500000001)
        (0.5 + 545439823.201985j)
        >>> zetazero(1500000002)
        (0.5 + 545439823.325697j)

    This confirms the data given by J. van de Lune,
    H. J. J. te Riele and D. T. Winter in 1986.
    """
    if t < 14.1347251417347:
        return 0
    x = gram_index(ctx, t)
    k = int(ctx.floor(x))
    wpinitial = ctx.prec
    wpz, fp_tolerance = comp_fp_tolerance(ctx, k)
    ctx.prec = wpz
    a = ctx.siegelz(t)
    if k == -1 and a < 0:
        return 0
    elif k == -1 and a > 0:
        return 1
    if k + 2 < 400000000:
        Rblock = find_rosser_block_zero(ctx, k + 2)
    else:
        Rblock = search_supergood_block(ctx, k + 2, fp_tolerance)
    n1, n2 = Rblock[1]
    if n2 - n1 == 1:
        b = Rblock[3][0]
        if a * b > 0:
            ctx.prec = wpinitial
            return k + 1
        else:
            ctx.prec = wpinitial
            return k + 2
    my_zero_number, block, T, V = Rblock
    zero_number_block = n2 - n1
    T, V, separated = separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=ctx.inf, fp_tolerance=fp_tolerance)
    n = count_to(ctx, t, T, V)
    ctx.prec = wpinitial
    return n + n1 + 1