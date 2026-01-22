import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def to_integral_exact(self, a):
    """Rounds to an integer.

        When the operand has a negative exponent, the result is the same
        as using the quantize() operation using the given operand as the
        left-hand-operand, 1E+0 as the right-hand-operand, and the precision
        of the operand as the precision setting; Inexact and Rounded flags
        are allowed in this operation.  The rounding mode is taken from the
        context.

        >>> ExtendedContext.to_integral_exact(Decimal('2.1'))
        Decimal('2')
        >>> ExtendedContext.to_integral_exact(Decimal('100'))
        Decimal('100')
        >>> ExtendedContext.to_integral_exact(Decimal('100.0'))
        Decimal('100')
        >>> ExtendedContext.to_integral_exact(Decimal('101.5'))
        Decimal('102')
        >>> ExtendedContext.to_integral_exact(Decimal('-101.5'))
        Decimal('-102')
        >>> ExtendedContext.to_integral_exact(Decimal('10E+5'))
        Decimal('1.0E+6')
        >>> ExtendedContext.to_integral_exact(Decimal('7.89E+77'))
        Decimal('7.89E+77')
        >>> ExtendedContext.to_integral_exact(Decimal('-Inf'))
        Decimal('-Infinity')
        """
    a = _convert_other(a, raiseit=True)
    return a.to_integral_exact(context=self)