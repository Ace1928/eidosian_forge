from cvxpy.expressions.variable import Variable
def maximum_canon(expr, args):
    shape = expr.shape
    t = Variable(shape)
    constraints = [t >= elem for elem in args]
    return (t, constraints)