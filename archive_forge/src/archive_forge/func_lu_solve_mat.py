from copy import copy
from ..libmp.backend import xrange
def lu_solve_mat(ctx, a, b):
    """Solve a * x = b  where a and b are matrices."""
    r = ctx.matrix(a.rows, b.cols)
    for i in range(b.cols):
        c = ctx.lu_solve(a, b.column(i))
        for j in range(len(c)):
            r[j, i] = c[j]
    return r