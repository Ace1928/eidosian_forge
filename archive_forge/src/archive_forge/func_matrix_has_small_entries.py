import string
from ..sage_helper import _within_sage, sage_method
def matrix_has_small_entries(A, bound):
    if A.base_ring().is_exact():
        return A == 0
    else:
        return univ_matrix_norm(A) < bound