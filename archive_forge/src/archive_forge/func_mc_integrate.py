import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
def mc_integrate(lower_lim, upper_lim, nsamps):
    """
            approximate the definite integral of `func` from
            `lower_lim` to `upper_lim`
            """
    out = cuda.to_device(np.zeros(nsamps, dtype='float32'))
    rng_states = create_xoroshiro128p_states(nsamps, seed=42)
    mc_integrator_kernel.forall(nsamps)(out, rng_states, lower_lim, upper_lim)
    factor = (upper_lim - lower_lim) / (nsamps - 1)
    return sum_reduce(out) * factor