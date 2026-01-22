import re
import unittest
from oslo_config import types
def test_non_digits_are_invalid(self):
    self.assertInvalid('12a45')