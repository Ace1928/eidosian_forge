import scipy.sparse as sp
def constant_grad(expr):
    """Returns the gradient of constant terms in an expression.

    Matrix expressions are vectorized, so the gradient is a matrix.

    Args:
        expr: An expression.

    Returns:
        A map of variable value to empty SciPy CSC sparse matrices.
    """
    grad = {}
    for var in expr.variables():
        rows = var.size
        cols = expr.size
        if (rows, cols) == (1, 1):
            grad[var] = 0.0
        else:
            grad[var] = sp.csc_matrix((rows, cols), dtype='float64')
    return grad