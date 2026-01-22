import sys
from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase
def target_shared_assign(r, x):
    r[()] = x[()]