from functools import reduce
from operator import mul, add
def mat_dot_vec(iter_mat, iter_vec, iter_term=None):
    if iter_term is None:
        return [vec_dot_vec(row, iter_vec) for row in iter_mat]
    else:
        return [vec_dot_vec(row, iter_vec) + term for row, term in zip(iter_mat, iter_term)]