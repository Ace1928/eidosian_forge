from __future__ import annotations
import math
import re
import sys
from fractions import Fraction
from typing import TYPE_CHECKING
from .i18n import _gettext as _
from .i18n import _ngettext, decimal_separator, thousands_separator
from .i18n import _ngettext_noop as NS_
from .i18n import _pgettext as P_
def fractional(value: NumberOrString) -> str:
    """Convert to fractional number.

    There will be some cases where one might not want to show ugly decimal places for
    floats and decimals.

    This function returns a human-readable fractional number in form of fractions and
    mixed fractions.

    Pass in a string, or a number or a float, and this function returns:

    * a string representation of a fraction
    * or a whole number
    * or a mixed fraction
    * or the str output of the value, if it could not be converted

    Examples:
        ```pycon
        >>> fractional(0.3)
        '3/10'
        >>> fractional(1.3)
        '1 3/10'
        >>> fractional(float(1/3))
        '1/3'
        >>> fractional(1)
        '1'
        >>> fractional("ten")
        'ten'
        >>> fractional(None)
        'None'

        ```

    Args:
        value (int, float, str): Integer to convert.

    Returns:
        str: Fractional number as a string.
    """
    try:
        number = float(value)
        if not math.isfinite(number):
            return _format_not_finite(number)
    except (TypeError, ValueError):
        return str(value)
    whole_number = int(number)
    frac = Fraction(number - whole_number).limit_denominator(1000)
    numerator = frac.numerator
    denominator = frac.denominator
    if whole_number and (not numerator) and (denominator == 1):
        return f'{whole_number:.0f}'
    if not whole_number:
        return f'{numerator:.0f}/{denominator:.0f}'
    return f'{whole_number:.0f} {numerator:.0f}/{denominator:.0f}'