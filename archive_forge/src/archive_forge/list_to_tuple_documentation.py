from pythran.analyses import Aliases, FixedSizeList
from pythran.tables import MODULES
from pythran.passmanager import Transformation
from pythran.utils import path_to_attr
import gast as ast

        Replace list calls by static_list calls when possible

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(n):\n"
        ...                  "    x = builtins.list(n)\n"
        ...                  "    x[0] = 0\n"
        ...                  "    return builtins.tuple(x)")
        >>> pm = passmanager.PassManager("test")
        >>> _, node = pm.apply(ListToTuple, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(n):
            x = builtins.pythran.static_list(n)
            x[0] = 0
            return builtins.tuple(x)

        >>> node = ast.parse("def foo(n):\n"
        ...                  "    x = builtins.list(n)\n"
        ...                  "    x[0] = 0\n"
        ...                  "    return x")
        >>> pm = passmanager.PassManager("test")
        >>> _, node = pm.apply(ListToTuple, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(n):
            x = builtins.list(n)
            x[0] = 0
            return x
        