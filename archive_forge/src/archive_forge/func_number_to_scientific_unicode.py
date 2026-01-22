from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def number_to_scientific_unicode(number, uncertainty=None, unit=None, fmt=None):
    u"""Formats a number as unicode (optionally with unit/uncertainty)

    Parameters
    ----------
    number : float (w or w/o unit)
    uncertainty : same as number
    unit : unit
    fmt : int or callable

    Examples
    --------
    >>> number_to_scientific_unicode(3.14) == u'3.14'
    True
    >>> number_to_scientific_unicode(3.14159265e-7) == u'3.1416·10⁻⁷'
    True
    >>> import quantities as pq
    >>> number_to_scientific_unicode(2**0.5 * pq.m / pq.s)
    '1.4142 m/s'

    """
    return _number_to_X(number, uncertainty, unit, fmt, unicode_of_unit, _unicode_pow_10)