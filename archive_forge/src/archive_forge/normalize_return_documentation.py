from pythran.analyses import CFG, YieldPoints
from pythran.passmanager import Transformation
import gast as ast

    Adds Return statement when they are implicit,
    and adds the None return value when not set

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(y): print(y)")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeReturn, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(y):
        print(y)
        return builtins.None
    