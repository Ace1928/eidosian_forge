import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def ln(self, a):
    """Returns the natural (base e) logarithm of the operand.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.ln(Decimal('0'))
        Decimal('-Infinity')
        >>> c.ln(Decimal('1.000'))
        Decimal('0')
        >>> c.ln(Decimal('2.71828183'))
        Decimal('1.00000000')
        >>> c.ln(Decimal('10'))
        Decimal('2.30258509')
        >>> c.ln(Decimal('+Infinity'))
        Decimal('Infinity')
        >>> c.ln(1)
        Decimal('0')
        """
    a = _convert_other(a, raiseit=True)
    return a.ln(context=self)