import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_run_pass_event(self):

    @njit
    def foo(x):
        return x + x
    with ev.install_recorder('numba:run_pass') as recorder:
        foo(2)
    self.assertGreater(len(recorder.buffer), 0)
    for _, event in recorder.buffer:
        data = event.data
        self.assertIsInstance(data['name'], str)
        self.assertIsInstance(data['qualname'], str)
        self.assertIsInstance(data['module'], str)
        self.assertIsInstance(data['flags'], str)
        self.assertIsInstance(data['args'], str)
        self.assertIsInstance(data['return_type'], str)