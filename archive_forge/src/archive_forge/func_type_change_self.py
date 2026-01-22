from numba import cuda
@cuda.jit
def type_change_self(x, y):
    if x > 1 and y > 0:
        return x + type_change_self(x - y, y)
    else:
        return y