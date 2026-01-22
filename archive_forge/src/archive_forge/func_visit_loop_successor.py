import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def visit_loop_successor(self, node):
    for successor in self.cfg.successors(node):
        if successor is not node.body[0]:
            if isinstance(node, ast.While):
                bound_range(self.result, self.aliases, ast.UnaryOp(ast.Not(), node.test))
            return [successor]