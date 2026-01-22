import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def compare_total_mag(self, a, b):
    """Compares two operands using their abstract representation ignoring sign.

        Like compare_total, but with operand's sign ignored and assumed to be 0.
        """
    a = _convert_other(a, raiseit=True)
    return a.compare_total_mag(b)