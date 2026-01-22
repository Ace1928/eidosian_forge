from numba import cuda
@cuda.jit(device=True)
def runaway_self(x):
    return runaway_self(x)