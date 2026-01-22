from .._util import get_backend
def pseudo_rev(t, kf, kb, prod, major, minor, backend=None):
    """Analytic product transient of a reversible pseudo first order reaction.

    Product concentration vs time from pseudo-first order reversible kinetics.

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
    return (-kb * prod + kf * major * minor + (kb * prod - kf * major * minor) * be.exp(-t * (kb + kf * major))) / (kb + kf * major)