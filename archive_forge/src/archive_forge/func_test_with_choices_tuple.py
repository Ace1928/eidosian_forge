import re
import unittest
from oslo_config import types
def test_with_choices_tuple(self):
    t = types.Port(choices=(80, 457))
    self._test_with_choices(t)