import re
import unittest
from oslo_config import types
def test_no_mapping_produces_error(self):
    self.assertInvalid('foo,bar')