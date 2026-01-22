from pythran.analyses import Ancestors
from pythran.passmanager import Transformation
import gast as ast


    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("""
    ... def foo(y):
    ...  if y: return 1
    ...  return 2""")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeIfElse, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(y):
        if y:
            return 1
        else:
            return 2

    >>> node = ast.parse("""
    ... def foo(y):
    ...  if y:
    ...    z = y + 1
    ...    if z:
    ...      return 1
    ...    else:
    ...      return 3
    ...  return 2""")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeIfElse, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(y):
        if y:
            z = (y + 1)
            if z:
                return 1
            else:
                return 3
        else:
            return 2
    