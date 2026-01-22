from pythran.analyses import RangeValues
from pythran.passmanager import Transformation
import gast as ast
from math import isinf
from copy import deepcopy
def visit_range(self, node):
    range_value = self.range_values[node]
    if isinf(range_value.high):
        return self.generic_visit(node)
    elif range_value.low == range_value.high:
        self.update = True
        return ast.Constant(range_value.low, None)
    else:
        return self.generic_visit(node)