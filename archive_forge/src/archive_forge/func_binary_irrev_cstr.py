from .._util import get_backend
def binary_irrev_cstr(t, k, r, p, fr, fp, fv, n=1, backend=None):
    """Analytic solution for ``2 A -> n B`` in a CSTR.

    Parameters
    ----------
    t : array_like
    k : float_like
        Rate constant
    r : float_like
        Initial concentration of reactant.
    p : float_like
        Initial concentration of product.
    fr : float_like
        Concentration of reactant in feed.
    fp : float_like
        Concentration of product in feed.
    fv : float_like
        Feed rate / tank volume ratio.
    n : int
    backend : module or str
        Default is 'numpy', can also be e.g. ``sympy``.

    Returns
    -------
    length-2 tuple
        concentrations of reactant and product

    """
    be = get_backend(backend)
    atanh = getattr(be, 'atanh', be.arctanh)
    three = 3 * be.cos(0)
    x0 = 1 / k
    x1 = be.sqrt(fv)
    x2 = 8 * k
    x3 = fr * x2
    x4 = be.sqrt(fv + x3)
    x5 = x1 * x4
    x6 = x1 * x4 / 2
    x7 = atanh((-fv ** (three / 2) * x4 - 4 * k * r * x5) / (fv ** 2 + fv * x3))
    x8 = fv * t
    x9 = fp * x2
    x10 = 4 * k * n
    x11 = fr * x10
    x12 = be.exp(x8)
    x13 = n * x12
    return (x0 * (-fv + x5 * be.tanh(t * x6 - x7)) / 4, x0 * (fv * x13 + 8 * k * p + r * x10 - x1 * x13 * x4 * be.tanh(x6 * (t - 2 * x7 / (x1 * x4))) + x11 * x12 - x11 + x12 * x9 - x9) * be.exp(-x8) / 8)