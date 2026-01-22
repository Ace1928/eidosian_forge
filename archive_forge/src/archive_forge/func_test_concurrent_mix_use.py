import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def test_concurrent_mix_use(self):
    self.run_compile([self.run_jit(nopython=True, cache=True), self.run_jit(nopython=True), self.run_vectorize(nopython=True, cache=True), self.run_vectorize(nopython=True), self.run_guvectorize(nopython=True, cache=True), self.run_guvectorize(nopython=True)])