import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
def guard_args(val: int):
    return