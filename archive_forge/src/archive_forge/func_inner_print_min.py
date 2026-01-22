from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
def inner_print_min(args):
    if len(args) == 1:
        return self._print(args[0])
    half = len(args) // 2
    return '((%(a)s < %(b)s) ? %(a)s : %(b)s)' % {'a': inner_print_min(args[:half]), 'b': inner_print_min(args[half:])}