import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def min_mag(self, a, b):
    """Compares the values numerically with their sign ignored.

        >>> ExtendedContext.min_mag(Decimal('3'), Decimal('-2'))
        Decimal('-2')
        >>> ExtendedContext.min_mag(Decimal('-3'), Decimal('NaN'))
        Decimal('-3')
        >>> ExtendedContext.min_mag(1, -2)
        Decimal('1')
        >>> ExtendedContext.min_mag(Decimal(1), -2)
        Decimal('1')
        >>> ExtendedContext.min_mag(1, Decimal(-2))
        Decimal('1')
        """
    a = _convert_other(a, raiseit=True)
    return a.min_mag(b, context=self)