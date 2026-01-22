import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
@classmethod
def sympy_expression_namespace(cls, expr):
    """Traverses the (func, args) tree of an expression and creates a SymPy
        namespace. All other modules are imported only as a module name. That way
        the namespace is not polluted and rests quite small. It probably causes much
        more variable lookups and so it takes more time, but there are no tests on
        that for the moment."""
    if expr is None:
        return {}
    else:
        funcname = str(expr.func)
        regexlist = ["<class \\'sympy[\\w.]*?.([\\w]*)\\'>$", '<function ([\\w]*) at 0x[\\w]*>$']
        for r in regexlist:
            m = re.match(r, funcname)
            if m is not None:
                funcname = m.groups()[0]
        args_dict = {}
        for a in expr.args:
            if isinstance(a, Symbol) or isinstance(a, NumberSymbol) or a in [I, zoo, oo]:
                continue
            else:
                args_dict.update(cls.sympy_expression_namespace(a))
        args_dict.update({funcname: expr.func})
        return args_dict