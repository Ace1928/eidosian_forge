import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def function_visitor(self, node):
    parent_result = self.result
    self.result = defaultdict(lambda: UNKNOWN_RANGE, [(k, v) for k, v in parent_result.items() if isinstance(k, ast.FunctionDef)])
    try:
        self.no_backward = 0
        self.no_if_split = 0
        self.cfg_visit(next(self.cfg.successors(node)))
        for k, v in self.result.items():
            parent_result[k] = v
        self.result = parent_result
    except RangeValueTooCostly:
        self.result = parent_result
        rvs = RangeValuesSimple(self)
        rvs.visit(node)