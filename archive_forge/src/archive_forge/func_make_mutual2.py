from numba import jit
def make_mutual2(jit=lambda x: x):

    @jit
    def foo(x):
        if x > 0:
            return 2 * bar(z=1, y=x)
        return 1 + x

    @jit
    def bar(y, z):
        return foo(x=y - z)
    return (foo, bar)