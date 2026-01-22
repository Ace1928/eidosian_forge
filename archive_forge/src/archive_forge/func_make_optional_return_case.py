from numba import cuda
def make_optional_return_case(jit=lambda x: x):

    @jit
    def foo(x):
        if x > 5:
            return x - 1
        else:
            return

    @jit
    def bar(x):
        out = foo(x)
        if out is None:
            return out
        elif out < 8:
            return out
        else:
            return x * bar(out)
    return bar