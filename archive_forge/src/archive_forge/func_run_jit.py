import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def run_jit(self, **options):

    def runner():
        cfunc = jit(**options)(foo)
        return cfunc(4, 10)
    return runner