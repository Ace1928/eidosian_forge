from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int
def letter_form_to_array_form(array_form, group):
    """
    This method converts a list given with possible repetitions of elements in
    it. It returns a new list such that repetitions of consecutive elements is
    removed and replace with a tuple element of size two such that the first
    index contains `value` and the second index contains the number of
    consecutive repetitions of `value`.

    """
    a = list(array_form[:])
    new_array = []
    n = 1
    symbols = group.symbols
    for i in range(len(a)):
        if i == len(a) - 1:
            if a[i] == a[i - 1]:
                if -a[i] in symbols:
                    new_array.append((-a[i], -n))
                else:
                    new_array.append((a[i], n))
            elif -a[i] in symbols:
                new_array.append((-a[i], -1))
            else:
                new_array.append((a[i], 1))
            return new_array
        elif a[i] == a[i + 1]:
            n += 1
        else:
            if -a[i] in symbols:
                new_array.append((-a[i], -n))
            else:
                new_array.append((a[i], n))
            n = 1