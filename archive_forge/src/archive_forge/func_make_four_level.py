from numba import jit
def make_four_level(jit=lambda x: x):

    @jit
    def first(x):
        if x > 0:
            return second(x) * 2
        else:
            return 1

    @jit
    def second(x):
        return third(x) * 3

    @jit
    def third(x):
        return fourth(x) * 4

    @jit
    def fourth(x):
        return first(x / 2 - 1)
    return first