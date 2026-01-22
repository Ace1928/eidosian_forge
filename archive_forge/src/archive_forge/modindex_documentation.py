from pythran.analyses import UseDefChains, Ancestors, Aliases, RangeValues
from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
from pythran.tables import MODULES
import gast as ast
from copy import deepcopy

    Simplify modulo on loop index

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> pm = passmanager.PassManager("test")
    >>> code = """
    ... def foo(x):
    ...     y = builtins.len(x)
    ...     for i in builtins.range(8):
    ...         z = i % y"""
    >>> node = ast.parse(code)
    >>> _, node = pm.apply(ModIndex, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(x):
        y = builtins.len(x)
        i_m = ((0 - 1) % y)
        for i in builtins.range(8):
            i_m = (0 if ((i_m + 1) == y) else (i_m + 1))
            z = i_m
    