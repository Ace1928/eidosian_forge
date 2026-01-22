import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def tree2str_translate(self, tree):
    """Converts a tree to string with translations.

        Explanation
        ===========

        Function names are translated by translate_func.
        Other strings are translated by translate_str.
        """
    if isinstance(tree, str):
        return self.translate_str(tree)
    elif isinstance(tree, tuple) and len(tree) == 2:
        return self.translate_func(tree[0][:-1], tree[1])
    else:
        return ''.join([self.tree2str_translate(t) for t in tree])