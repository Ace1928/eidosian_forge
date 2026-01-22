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
def eliminate_words(self, words, _all=False, inverse=True):
    """
        Replace each subword from the dictionary `words` by words[subword].
        If words is a list, replace the words by the identity.

        """
    again = True
    new = self
    if isinstance(words, dict):
        while again:
            again = False
            for sub in words:
                prev = new
                new = new.eliminate_word(sub, words[sub], _all=_all, inverse=inverse)
                if new != prev:
                    again = True
    else:
        while again:
            again = False
            for sub in words:
                prev = new
                new = new.eliminate_word(sub, _all=_all, inverse=inverse)
                if new != prev:
                    again = True
    return new