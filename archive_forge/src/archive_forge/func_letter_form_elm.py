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
@property
def letter_form_elm(self):
    """
        """
    group = self.group
    r = self.letter_form
    return [group.dtype(((elm, 1),)) if elm.is_Symbol else group.dtype(((-elm, -1),)) for elm in r]