from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def irls_units(x, y, **kwargs):
    """Units aware version of :func:`irls`

    Parameters
    ----------
    x : array_like
    y : array_like
    \\*\\*kwargs
        Keyword arguments passed on to :func:`irls`

    """
    x_unit, y_unit = (unit_of(x), unit_of(y))
    x_ul, y_ul = (to_unitless(x, x_unit), to_unitless(y, y_unit))
    beta, vcv, info = irls(x_ul, y_ul, **kwargs)
    beta_tup = _beta_tup(beta, x_unit, y_unit)
    return (beta_tup, vcv, info)