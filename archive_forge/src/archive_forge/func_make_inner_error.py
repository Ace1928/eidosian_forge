from numba import jit
def make_inner_error(jit=lambda x: x):

    @jit
    def outer(x):
        if x > 0:
            return inner(x)
        else:
            return 1

    @jit
    def inner(x):
        if x > 0:
            return outer(x - 1)
        else:
            return error_fun(x)

    @jit
    def error_fun(x):
        return x.ndim
    return outer