import re
import unittest
from oslo_config import types
def test_repr_with_choices(self):
    t = types.Port(choices=[80, 457])
    self.assertEqual('Port(choices=[80, 457])', repr(t))