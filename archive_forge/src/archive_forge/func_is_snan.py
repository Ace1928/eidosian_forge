import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_snan(self, a):
    """Return True if the operand is a signaling NaN;
        otherwise return False.

        >>> ExtendedContext.is_snan(Decimal('2.50'))
        False
        >>> ExtendedContext.is_snan(Decimal('NaN'))
        False
        >>> ExtendedContext.is_snan(Decimal('sNaN'))
        True
        >>> ExtendedContext.is_snan(1)
        False
        """
    a = _convert_other(a, raiseit=True)
    return a.is_snan()