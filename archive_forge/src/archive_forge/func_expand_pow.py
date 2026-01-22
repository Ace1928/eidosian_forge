from pythran.passmanager import Transformation
from pythran.analyses.ast_matcher import ASTMatcher, AST_any
from pythran.conversion import mangle
from pythran.utils import isnum
import gast as ast
import copy
def expand_pow(self, node, n):
    if n == 0:
        return ast.Constant(1, None)
    elif n == 1:
        return node
    else:
        node_square = self.replace(node)
        node_pow = self.expand_pow(node_square, n >> 1)
        if n & 1:
            return ast.BinOp(node_pow, ast.Mult(), copy.deepcopy(node))
        else:
            return node_pow