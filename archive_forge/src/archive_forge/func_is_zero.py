import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_zero(self, a):
    """Return True if the operand is a zero; otherwise return False.

        >>> ExtendedContext.is_zero(Decimal('0'))
        True
        >>> ExtendedContext.is_zero(Decimal('2.50'))
        False
        >>> ExtendedContext.is_zero(Decimal('-0E+2'))
        True
        >>> ExtendedContext.is_zero(1)
        False
        >>> ExtendedContext.is_zero(0)
        True
        """
    a = _convert_other(a, raiseit=True)
    return a.is_zero()