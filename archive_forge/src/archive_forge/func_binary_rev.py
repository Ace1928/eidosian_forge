from .._util import get_backend
def binary_rev(t, kf, kb, prod, major, minor, backend=None):
    """Analytic product transient of a reversible 2-to-1 reaction.

    Product concentration vs time from second order reversible kinetics.

    Parameters
    ----------
    t : float, Symbol or array_like
        Time.
    kf : number or Symbol
        Forward (bimolecular) rate constant.
    kb : number or Symbol
        Backward (unimolecular) rate constant.
    prod : number or Symbol
        Initial concentration of the complex.
    major : number or Symbol
        Initial concentration of the more abundant reactant.
    minor : number or Symbol
        Initial concentration of the less abundant reactant.
    backend : module or str
        Default is 'numpy', can also be e.g. ``sympy``.

    """
    be = get_backend(backend)
    X, Y, Z = (prod, major, minor)
    x0 = Y * kf
    x1 = Z * kf
    x2 = 2 * X * kf
    x3 = -kb - x0 - x1
    x4 = -x2 + x3
    x5 = be.sqrt(-4 * kf * (X ** 2 * kf + X * x0 + X * x1 + Z * x0) + x4 ** 2)
    x6 = kb + x0 + x1 + x5
    x7 = (x3 + x5) * be.exp(-t * x5)
    x8 = x3 - x5
    return (x4 * x8 + x5 * x8 + x7 * (x2 + x6)) / (2 * kf * (x6 + x7))