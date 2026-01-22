from numba import jit
@jit(nopython=True)
def runaway_mutual(x):
    return runaway_mutual_inner(x)