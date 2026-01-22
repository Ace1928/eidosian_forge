import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def run_guvectorize(self, **options):

    def runner():
        sig = ['(f4, f4, f4[:])']
        cfunc = guvectorize(sig, '(),()->()', **options)(gufunc_foo)
        a = b = np.random.random(10).astype(np.float32)
        return cfunc(a, b)
    return runner