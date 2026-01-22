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
def scientific(value: NumberOrString, precision: int=2) -> str:
    """Return number in string scientific notation z.wq x 10ⁿ.

    Examples:
        ```pycon
        >>> scientific(float(0.3))
        '3.00 x 10⁻¹'
        >>> scientific(int(500))
        '5.00 x 10²'
        >>> scientific(-1000)
        '-1.00 x 10³'
        >>> scientific(1000, 1)
        '1.0 x 10³'
        >>> scientific(1000, 3)
        '1.000 x 10³'
        >>> scientific("99")
        '9.90 x 10¹'
        >>> scientific("foo")
        'foo'
        >>> scientific(None)
        'None'

        ```

    Args:
        value (int, float, str): Input number.
        precision (int): Number of decimal for first part of the number.

    Returns:
        str: Number in scientific notation z.wq x 10ⁿ.
    """
    exponents = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
    try:
        value = float(value)
        if not math.isfinite(value):
            return _format_not_finite(value)
    except (ValueError, TypeError):
        return str(value)
    fmt = '{:.%se}' % str(int(precision))
    n = fmt.format(value)
    part1, part2 = n.split('e')
    part2 = re.sub('^\\+?(\\-?)0*(.+)$', '\\1\\2', part2)
    new_part2 = []
    for char in part2:
        new_part2.append(exponents[char])
    final_str = part1 + ' x 10' + ''.join(new_part2)
    return final_str