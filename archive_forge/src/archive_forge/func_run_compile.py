import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def run_compile(self, fnlist):
    self._cache_dir = temp_directory(self.__class__.__name__)
    with override_config('CACHE_DIR', self._cache_dir):

        def chooser():
            for _ in range(10):
                fn = random.choice(fnlist)
                fn()
        ths = [threading.Thread(target=chooser) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()