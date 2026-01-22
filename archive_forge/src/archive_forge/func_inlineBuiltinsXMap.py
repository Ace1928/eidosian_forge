from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def inlineBuiltinsXMap(self, node):
    self.update = True
    elts = []
    nelts = min((len(n.elts) for n in node.args[1:]))
    for i in range(nelts):
        elts.append([n.elts[i] for n in node.args[1:]])
    return ast.List([ast.Call(node.args[0], elt, []) for elt in elts], ast.Load())