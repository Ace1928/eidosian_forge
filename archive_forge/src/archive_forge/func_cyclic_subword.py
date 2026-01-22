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
def cyclic_subword(self, from_i, to_j):
    group = self.group
    l = len(self)
    letter_form = self.letter_form
    period1 = int(from_i / l)
    if from_i >= l:
        from_i -= l * period1
        to_j -= l * period1
    diff = to_j - from_i
    word = letter_form[from_i:to_j]
    period2 = int(to_j / l) - 1
    word += letter_form * period2 + letter_form[:diff - l + from_i - l * period2]
    word = letter_form_to_array_form(word, group)
    return group.dtype(word)