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
def subword(self, from_i, to_j, strict=True):
    """
        For an associative word `self` and two positive integers `from_i` and
        `to_j`, `subword` returns the subword of `self` that begins at position
        `from_i` and ends at `to_j - 1`, indexing is done with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.subword(2, 6)
        a**3*b

        """
    group = self.group
    if not strict:
        from_i = max(from_i, 0)
        to_j = min(len(self), to_j)
    if from_i < 0 or to_j > len(self):
        raise ValueError('`from_i`, `to_j` must be positive and no greater than the length of associative word')
    if to_j <= from_i:
        return group.identity
    else:
        letter_form = self.letter_form[from_i:to_j]
        array_form = letter_form_to_array_form(letter_form, group)
        return group.dtype(array_form)