import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_infinite(self, a):
    """Return True if the operand is infinite; otherwise return False.

        >>> ExtendedContext.is_infinite(Decimal('2.50'))
        False
        >>> ExtendedContext.is_infinite(Decimal('-Inf'))
        True
        >>> ExtendedContext.is_infinite(Decimal('NaN'))
        False
        >>> ExtendedContext.is_infinite(1)
        False
        """
    a = _convert_other(a, raiseit=True)
    return a.is_infinite()