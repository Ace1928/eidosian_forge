from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def least_squares_units(x, y, w=1):
    """Units-aware least-squares (w or w/o weights) fit to data series.

    Parameters
    ----------
    x : array_like
    y : array_like
    w : array_like, optional

    """
    x_unit, y_unit = (unit_of(x), unit_of(y))
    integer_one = 1
    explicit_errors = w is not integer_one
    if explicit_errors:
        if unit_of(w) == y_unit ** (-2):
            _w = to_unitless(w, y_unit ** (-2))
        elif unit_of(w) == unit_of(1):
            _w = w
        else:
            raise ValueError('Incompatible units in y and w')
    else:
        _w = 1
    _x = to_unitless(x, x_unit)
    _y = to_unitless(y, y_unit)
    beta, vcv, r2 = least_squares(_x, _y, _w)
    beta_tup = _beta_tup(beta, x_unit, y_unit)
    return (beta_tup, vcv, float(r2))