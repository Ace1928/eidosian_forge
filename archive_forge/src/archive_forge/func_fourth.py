from numba import jit
@jit
def fourth(x):
    return first(x / 2 - 1)