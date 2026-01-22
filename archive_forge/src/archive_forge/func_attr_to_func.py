from pythran.analyses import Globals, Ancestors
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import attributes, functions, methods, MODULES
from pythran.tables import duplicated_methods
from pythran.conversion import mangle, demangle
from pythran.utils import isstr
import gast as ast
from functools import reduce
def attr_to_func(self, node, mod=None):
    if mod is None:
        mod = methods[node.attr][0]
    self.to_import.add(mangle(mod[0]))
    func = reduce(lambda v, o: ast.Attribute(v, o, ast.Load()), mod[1:] + (node.attr,), ast.Name(mangle(mod[0]), ast.Load(), None, None))
    return func