from cvxpy.expressions.variable import Variable
def cumsum_canon(expr, args):
    """Cumulative sum.
    """
    X = args[0]
    axis = expr.axis
    Y = Variable(expr.shape)
    if axis == 0:
        if expr.shape[0] == 1:
            return (X, [])
        else:
            constr = [X[1:] == Y[1:] - Y[:-1], Y[0] == X[0]]
    elif expr.shape[1] == 1:
        return (X, [])
    else:
        constr = [X[:, 1:] == Y[:, 1:] - Y[:, :-1], Y[:, 0] == X[:, 0]]
    return (Y, constr)