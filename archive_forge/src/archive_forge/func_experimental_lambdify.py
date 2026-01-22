import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def experimental_lambdify(*args, **kwargs):
    l = Lambdifier(*args, **kwargs)
    return l