import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def isintegral(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, bool))