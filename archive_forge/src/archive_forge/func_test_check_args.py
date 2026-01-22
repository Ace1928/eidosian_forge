import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
def test_check_args(self):
    with self.assertRaises(self._exception_type):
        guard_args(float(1.2))