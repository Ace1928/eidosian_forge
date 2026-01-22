import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def copy_decimal(self, a):
    """Returns a copy of the decimal object.

        >>> ExtendedContext.copy_decimal(Decimal('2.1'))
        Decimal('2.1')
        >>> ExtendedContext.copy_decimal(Decimal('-1.00'))
        Decimal('-1.00')
        >>> ExtendedContext.copy_decimal(1)
        Decimal('1')
        """
    a = _convert_other(a, raiseit=True)
    return Decimal(a)