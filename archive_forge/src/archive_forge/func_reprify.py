from __future__ import annotations
from typing import Any
from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str
from .printer import Printer, print_function
def reprify(self, args, sep):
    """
        Prints each item in `args` and joins them with `sep`.
        """
    return sep.join([self.doprint(item) for item in args])