import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def negate(node):
    if isinstance(node, ast.Name):
        raise UnsupportedExpression()
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Invert):
            return ast.Compare(node.operand, [ast.Eq()], [ast.Constant(-1, None)])
        if isinstance(node.op, ast.Not):
            return node.operand
        if isinstance(node.op, ast.UAdd):
            return node.operand
        if isinstance(node.op, ast.USub):
            return node.operand
    if isinstance(node, ast.BoolOp):
        new_values = [ast.UnaryOp(ast.Not(), v) for v in node.values]
        if isinstance(node.op, ast.Or):
            return ast.BoolOp(ast.And(), new_values)
        if isinstance(node.op, ast.And):
            return ast.BoolOp(ast.Or(), new_values)
    if isinstance(node, ast.Compare):
        cmps = [ast.Compare(x, [negate(o)], [y]) for x, o, y in zip([node.left] + node.comparators[:-1], node.ops, node.comparators)]
        if len(cmps) == 1:
            return cmps[0]
        return ast.BoolOp(ast.Or(), cmps)
    if isinstance(node, ast.Eq):
        return ast.NotEq()
    if isinstance(node, ast.NotEq):
        return ast.Eq()
    if isinstance(node, ast.Gt):
        return ast.LtE()
    if isinstance(node, ast.GtE):
        return ast.Lt()
    if isinstance(node, ast.Lt):
        return ast.GtE()
    if isinstance(node, ast.LtE):
        return ast.Gt()
    if isinstance(node, ast.In):
        return ast.NotIn()
    if isinstance(node, ast.NotIn):
        return ast.In()
    if isinstance(node, ast.Attribute):
        if node.attr == 'False':
            return ast.Constant(True, None)
        if node.attr == 'True':
            return ast.Constant(False, None)
    raise UnsupportedExpression()