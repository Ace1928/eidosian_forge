from cvxpy.expressions.constants import Constant
def indicator_canon(expr, args):
    return (Constant(0), args)