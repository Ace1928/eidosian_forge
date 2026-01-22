from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast
from functools import reduce
from collections import OrderedDict
from copy import deepcopy
def traverse_tuples(self, node, state, renamings):
    if isinstance(node, ast.Name):
        if state:
            renamings[node.id] = state
            self.update = True
    elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        [self.traverse_tuples(n, state + (i,), renamings) for i, n in enumerate(node.elts)]
    elif isinstance(node, (ast.Subscript, ast.Attribute)):
        if state:
            renamings[node] = state
            self.update = True
    else:
        raise NotImplementedError