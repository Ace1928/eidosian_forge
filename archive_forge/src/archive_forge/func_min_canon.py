from cvxpy.atoms.max import max
from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon
def min_canon(expr, args):
    axis = expr.axis
    del expr
    assert len(args) == 1
    tmp = max(-args[0], axis=axis)
    canon, constr = max_canon(tmp, tmp.args)
    return (-canon, constr)