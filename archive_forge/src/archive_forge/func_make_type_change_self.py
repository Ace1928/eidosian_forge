from numba import jit
def make_type_change_self(jit=lambda x: x):

    @jit
    def type_change_self(x, y):
        if x > 1 and y > 0:
            return x + type_change_self(x - y, y)
        else:
            return y
    return type_change_self