from pythran.analyses import ImportedIds
from pythran.passmanager import Transformation
import pythran.metadata as metadata
import gast as ast

    Turns multiple compare into a function with proper temporaries.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(a): return 0 < a + 1 < 3")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeCompare, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(a):
        return foo_compare0(a)
    def foo_compare0(a):
        $1 = (a + 1)
        if (0 < $1):
            pass
        else:
            return False
        if ($1 < 3):
            pass
        else:
            return False
        return True
    