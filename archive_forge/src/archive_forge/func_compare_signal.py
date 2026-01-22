import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def compare_signal(self, a, b):
    """Compares the values of the two operands numerically.

        It's pretty much like compare(), but all NaNs signal, with signaling
        NaNs taking precedence over quiet NaNs.

        >>> c = ExtendedContext
        >>> c.compare_signal(Decimal('2.1'), Decimal('3'))
        Decimal('-1')
        >>> c.compare_signal(Decimal('2.1'), Decimal('2.1'))
        Decimal('0')
        >>> c.flags[InvalidOperation] = 0
        >>> print(c.flags[InvalidOperation])
        0
        >>> c.compare_signal(Decimal('NaN'), Decimal('2.1'))
        Decimal('NaN')
        >>> print(c.flags[InvalidOperation])
        1
        >>> c.flags[InvalidOperation] = 0
        >>> print(c.flags[InvalidOperation])
        0
        >>> c.compare_signal(Decimal('sNaN'), Decimal('2.1'))
        Decimal('NaN')
        >>> print(c.flags[InvalidOperation])
        1
        >>> c.compare_signal(-1, 2)
        Decimal('-1')
        >>> c.compare_signal(Decimal(-1), 2)
        Decimal('-1')
        >>> c.compare_signal(-1, Decimal(2))
        Decimal('-1')
        """
    a = _convert_other(a, raiseit=True)
    return a.compare_signal(b, context=self)