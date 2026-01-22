from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
def cast_nocheck(self, value):
    """ Casts without checking if out of bounds or subnormal. """
    from sympy.functions import re, im
    return super().cast_nocheck(re(value)) + super().cast_nocheck(im(value)) * 1j