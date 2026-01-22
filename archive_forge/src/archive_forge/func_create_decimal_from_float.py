import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def create_decimal_from_float(self, f):
    """Creates a new Decimal instance from a float but rounding using self
        as the context.

        >>> context = Context(prec=5, rounding=ROUND_DOWN)
        >>> context.create_decimal_from_float(3.1415926535897932)
        Decimal('3.1415')
        >>> context = Context(prec=5, traps=[Inexact])
        >>> context.create_decimal_from_float(3.1415926535897932)
        Traceback (most recent call last):
            ...
        decimal.Inexact: None

        """
    d = Decimal.from_float(f)
    return d._fix(self)