from .functions import defun, defun_wrapped
def sure_number_block(ctx, n):
    """The number of good Rosser blocks needed to apply
    Turing method
    References:
    R. P. Brent, On the Zeros of the Riemann Zeta Function
    in the Critical Strip, Math. Comp. 33 (1979) 1361--1372
    T. Trudgian, Improvements to Turing Method, Math. Comp."""
    if n < 9 * 10 ** 5:
        return 2
    g = ctx.grampoint(n - 100)
    lg = ctx._fp.ln(g)
    brent = 0.0061 * lg ** 2 + 0.08 * lg
    trudgian = 0.0031 * lg ** 2 + 0.11 * lg
    N = ctx.ceil(min(brent, trudgian))
    N = int(N)
    return N