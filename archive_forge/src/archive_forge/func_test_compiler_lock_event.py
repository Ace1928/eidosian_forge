import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_compiler_lock_event(self):

    @njit
    def foo(x):
        return x + x
    foo(1)
    md = foo.get_metadata(foo.signatures[0])
    lock_duration = md['timers']['compiler_lock']
    self.assertIsInstance(lock_duration, float)
    self.assertGreater(lock_duration, 0)