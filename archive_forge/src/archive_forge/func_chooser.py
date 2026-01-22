import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def chooser():
    for _ in range(10):
        fn = random.choice(fnlist)
        fn()