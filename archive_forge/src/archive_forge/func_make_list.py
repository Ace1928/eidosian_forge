from numba import cuda
@jit
def make_list(n):
    if n <= 0:
        return None
    return (n, make_list(n - 1))