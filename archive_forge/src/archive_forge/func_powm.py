from ..libmp.backend import xrange
def powm(ctx, A, r):
    """
        Computes `A^r = \\exp(A \\log r)` for a matrix `A` and complex
        number `r`.

        **Examples**

        Powers and inverse powers of a matrix::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> powm(A, 2)
            [ 63.0  20.0   69.0]
            [174.0  89.0  199.0]
            [164.0  48.0  179.0]
            >>> chop(powm(powm(A, 4), 1/4.))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(20)(powm)(A, -4), -1/4.)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> chop(powm(powm(A, 1+0.5j), 1/(1+0.5j)))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(5)(powm)(A, -1.5), -1/(1.5))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]

        A Fibonacci-generating matrix::

            >>> powm([[1,1],[1,0]], 10)
            [89.0  55.0]
            [55.0  34.0]
            >>> fib(10)
            55.0
            >>> powm([[1,1],[1,0]], 6.5)
            [(16.5166626964253 - 0.0121089837381789j)  (10.2078589271083 + 0.0195927472575932j)]
            [(10.2078589271083 + 0.0195927472575932j)  (6.30880376931698 - 0.0317017309957721j)]
            >>> (phi**6.5 - (1-phi)**6.5)/sqrt(5)
            (10.2078589271083 - 0.0195927472575932j)
            >>> powm([[1,1],[1,0]], 6.2)
            [ (14.3076953002666 - 0.008222855781077j)  (8.81733464837593 + 0.0133048601383712j)]
            [(8.81733464837593 + 0.0133048601383712j)  (5.49036065189071 - 0.0215277159194482j)]
            >>> (phi**6.2 - (1-phi)**6.2)/sqrt(5)
            (8.81733464837593 - 0.0133048601383712j)

        """
    A = ctx.matrix(A)
    r = ctx.convert(r)
    prec = ctx.prec
    try:
        ctx.prec += 10
        if ctx.isint(r):
            v = A ** int(r)
        elif ctx.isint(r * 2):
            y = int(r * 2)
            v = ctx.sqrtm(A) ** y
        else:
            v = ctx.expm(r * ctx.logm(A))
    finally:
        ctx.prec = prec
    v *= 1
    return v