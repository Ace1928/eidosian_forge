import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def translate_func(self, func_name, argtree):
    """Translate function names and the tree of arguments.

        Explanation
        ===========

        If the function name is not in the dictionaries of dict_tuple_fun then the
        function is surrounded by a float((...).evalf()).

        The use of float is necessary as np.<function>(sympy.Float(..)) raises an
        error."""
    if func_name in self.dict_fun:
        new_name = self.dict_fun[func_name]
        argstr = self.tree2str_translate(argtree)
        return new_name + '(' + argstr
    elif func_name in ['Eq', 'Ne']:
        op = {'Eq': '==', 'Ne': '!='}
        return '(lambda x, y: x {} y)({}'.format(op[func_name], self.tree2str_translate(argtree))
    else:
        template = '(%s(%s)).evalf(' if self.use_evalf else '%s(%s'
        if self.float_wrap_evalf:
            template = 'float(%s)' % template
        elif self.complex_wrap_evalf:
            template = 'complex(%s)' % template
        float_wrap_evalf = self.float_wrap_evalf
        complex_wrap_evalf = self.complex_wrap_evalf
        self.float_wrap_evalf = False
        self.complex_wrap_evalf = False
        ret = template % (func_name, self.tree2str_translate(argtree))
        self.float_wrap_evalf = float_wrap_evalf
        self.complex_wrap_evalf = complex_wrap_evalf
        return ret