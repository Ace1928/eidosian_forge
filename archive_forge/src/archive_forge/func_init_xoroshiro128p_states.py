import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
def init_xoroshiro128p_states(states, seed, subsequence_start=0, stream=0):
    """Initialize RNG states on the GPU for parallel generators.

    This initializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64
    random numbers, all of the RNG states produced by this function are
    guaranteed to be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    :type states: 1D DeviceNDArray, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type seed: uint64
    :param seed: starting seed for list of generators
    """
    states_cpu = np.empty(shape=states.shape, dtype=xoroshiro128p_dtype)
    init_xoroshiro128p_states_cpu(states_cpu, seed, subsequence_start)
    states.copy_to_device(states_cpu, stream=stream)