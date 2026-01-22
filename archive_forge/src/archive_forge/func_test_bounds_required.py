import re
import unittest
from oslo_config import types
def test_bounds_required(self):
    self.type_instance = types.Dict(types.String(), bounds=True)
    self.assertInvalid('foo:bar,baz:123')
    self.assertInvalid('{foo:bar,baz:123')
    self.assertInvalid('foo:bar,baz:123}')