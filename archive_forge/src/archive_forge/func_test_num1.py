import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_num1(self):
    self.check_values('num1', 1, [1, 2, 3, 4, 5, -1, -2, -3, -4, -5], [0, 6, -6, '0', '6', '-6', 0.0, 6.0, -6.0, [1], (1,), {1: 1}, None], [1, 2, 3, 4, 5, -1, -2, -3, -4, -5])