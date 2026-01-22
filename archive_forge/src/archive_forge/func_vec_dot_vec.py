from functools import reduce
from operator import mul, add
def vec_dot_vec(vec1, vec2):
    return reducemap((vec1, vec2), add, mul)