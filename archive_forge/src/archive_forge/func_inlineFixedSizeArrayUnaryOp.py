from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def inlineFixedSizeArrayUnaryOp(self, node):
    if isinstance(node.operand, (ast.Constant, ast.List, ast.Tuple)):
        return node
    base, size = self.fixedSizeArray(node.operand)
    if not base:
        return node
    self.update = True
    operands = [ast.UnaryOp(type(node.op)(), self.make_array_index(base, size, i)) for i in range(size)]
    res = ast.Call(path_to_attr(('numpy', 'array')), [ast.Tuple(operands, ast.Load())], [])
    self.aliases[res.func] = {path_to_node(('numpy', 'array'))}
    return res