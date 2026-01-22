import string
from ..sage_helper import _within_sage, sage_method
def first_square_submatrix(A):
    r, c = (A.nrows(), A.ncols())
    if r <= c:
        return A.matrix_from_columns(range(r))
    else:
        return A.matrix_from_rows(range(c))