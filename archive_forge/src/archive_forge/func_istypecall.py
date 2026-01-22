from pythran.passmanager import Transformation
import gast as ast
from functools import reduce
def istypecall(node):
    if not isinstance(node, ast.Call):
        return False
    return getattr(node.func, 'attr', None) == 'type'