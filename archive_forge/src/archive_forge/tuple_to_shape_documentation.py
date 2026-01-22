from pythran.analyses import Aliases
from pythran.tables import MODULES
from pythran.passmanager import Transformation
from pythran.utils import pythran_builtin_attr
import gast as ast

    Replace tuple nodes by shape when relevant

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(n): import numpy; return numpy.ones((n,4))")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(TupleToShape, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(n):
        import numpy
        return numpy.ones(builtins.pythran.make_shape(n, 4))
    