from cvxpy.atoms.affine.vec import vec
def explicit_sum(expr):
    x = vec(expr)
    summation = x[0]
    for xi in x[1:]:
        summation += xi
    return summation