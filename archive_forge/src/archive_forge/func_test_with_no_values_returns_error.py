import re
import unittest
from oslo_config import types
def test_with_no_values_returns_error(self):
    self.type_instance = types.String(choices=[])
    self.assertInvalid('foo')