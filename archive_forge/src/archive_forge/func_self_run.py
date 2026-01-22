import sys
import numpy as np
from numba import njit
from numba.tests.support import TestCase
def self_run():
    mod = sys.modules[__name__]
    _TestModule().run_module(mod)