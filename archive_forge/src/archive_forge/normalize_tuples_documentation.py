from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast
from functools import reduce
from collections import OrderedDict
from copy import deepcopy

    Remove implicit tuple -> variable conversion.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): a=(1,2.) ; i,j = a")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeTuples, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = (1, 2.0)
        i = a[0]
        j = a[1]
    