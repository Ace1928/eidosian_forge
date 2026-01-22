import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def get_dict_str(self):
    dict_str = dict(self.builtin_not_functions)
    if self.use_np:
        dict_str.update(self.numpy_not_functions)
    if self.use_python_math:
        dict_str.update(self.math_not_functions)
    if self.use_python_cmath:
        dict_str.update(self.cmath_not_functions)
    if self.use_interval:
        dict_str.update(self.interval_not_functions)
    return dict_str