import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def max_mag(self, a, b):
    """Compares the values numerically with their sign ignored.

        >>> ExtendedContext.max_mag(Decimal('7'), Decimal('NaN'))
        Decimal('7')
        >>> ExtendedContext.max_mag(Decimal('7'), Decimal('-10'))
        Decimal('-10')
        >>> ExtendedContext.max_mag(1, -2)
        Decimal('-2')
        >>> ExtendedContext.max_mag(Decimal(1), -2)
        Decimal('-2')
        >>> ExtendedContext.max_mag(1, Decimal(-2))
        Decimal('-2')
        """
    a = _convert_other(a, raiseit=True)
    return a.max_mag(b, context=self)