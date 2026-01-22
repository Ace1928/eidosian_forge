import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def translate_str(self, estr):
    """Translate substrings of estr using in order the dictionaries in
        dict_tuple_str."""
    for pattern, repl in self.dict_str.items():
        estr = re.sub(pattern, repl, estr)
    return estr