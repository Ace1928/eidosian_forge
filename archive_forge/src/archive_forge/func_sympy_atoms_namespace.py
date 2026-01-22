import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
@staticmethod
def sympy_atoms_namespace(expr):
    """For no real reason this function is separated from
        sympy_expression_namespace. It can be moved to it."""
    atoms = expr.atoms(Symbol, NumberSymbol, I, zoo, oo)
    d = {}
    for a in atoms:
        d[str(a)] = a
    return d