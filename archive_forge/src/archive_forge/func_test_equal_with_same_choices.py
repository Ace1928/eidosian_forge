import re
import unittest
from oslo_config import types
def test_equal_with_same_choices(self):
    t1 = types.Port(choices=[80, 457])
    t2 = types.Port(choices=[457, 80])
    t3 = types.Port(choices=(457, 80))
    t4 = types.Port(choices=[(457, 'ab'), (80, 'xy')])
    self.assertTrue(t1 == t2 == t3 == t4)