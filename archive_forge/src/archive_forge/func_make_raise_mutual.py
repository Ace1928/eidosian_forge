from numba import jit
def make_raise_mutual(jit=lambda x: x):

    @jit
    def outer(x):
        if x > 0:
            return inner(x)
        else:
            return 1

    @jit
    def inner(x):
        if x == 1:
            raise ValueError('raise_mutual')
        elif x > 0:
            return outer(x - 1)
        else:
            return 1
    return outer