import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def get_dict_fun(self):
    dict_fun = dict(self.builtin_functions_different)
    if self.use_np:
        for s in self.numpy_functions_same:
            dict_fun[s] = 'np.' + s
        for k, v in self.numpy_functions_different.items():
            dict_fun[k] = 'np.' + v
    if self.use_python_math:
        for s in self.math_functions_same:
            dict_fun[s] = 'math.' + s
        for k, v in self.math_functions_different.items():
            dict_fun[k] = 'math.' + v
    if self.use_python_cmath:
        for s in self.cmath_functions_same:
            dict_fun[s] = 'cmath.' + s
        for k, v in self.cmath_functions_different.items():
            dict_fun[k] = 'cmath.' + v
    if self.use_interval:
        for s in self.interval_functions_same:
            dict_fun[s] = 'imath.' + s
        for k, v in self.interval_functions_different.items():
            dict_fun[k] = 'imath.' + v
    return dict_fun