import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
@cuda.jit
def mc_integrator_kernel(out, rng_states, lower_lim, upper_lim):
    """
            kernel to draw random samples and evaluate the function to
            be integrated at those sample values
            """
    size = len(out)
    gid = cuda.grid(1)
    if gid < size:
        samp = xoroshiro128p_uniform_float32(rng_states, gid)
        samp = samp * (upper_lim - lower_lim) + lower_lim
        y = func(samp)
        out[gid] = y