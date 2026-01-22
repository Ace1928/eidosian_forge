from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def fixedSizeArray(self, node):
    if isinstance(node, ast.Constant):
        return (node, 1)
    if isinstance(node, (ast.List, ast.Tuple)):
        return (node, len(node.elts))
    if not isinstance(node, ast.Call):
        return (None, 0)
    func_aliases = self.aliases[node.func]
    if len(func_aliases) != 1:
        return (None, 0)
    obj = next(iter(func_aliases))
    if obj not in (MODULES['numpy']['array'], MODULES['numpy']['asarray']):
        return (None, 0)
    if len(node.args) != 1:
        return (None, 0)
    if isinstance(node.args[0], (ast.List, ast.Tuple)):
        return (node.args[0], len(node.args[0].elts))
    return (None, 0)