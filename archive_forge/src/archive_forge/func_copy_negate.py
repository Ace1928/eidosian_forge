import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def copy_negate(self, a):
    """Returns a copy of the operand with the sign inverted.

        >>> ExtendedContext.copy_negate(Decimal('101.5'))
        Decimal('-101.5')
        >>> ExtendedContext.copy_negate(Decimal('-101.5'))
        Decimal('101.5')
        >>> ExtendedContext.copy_negate(1)
        Decimal('-1')
        """
    a = _convert_other(a, raiseit=True)
    return a.copy_negate()