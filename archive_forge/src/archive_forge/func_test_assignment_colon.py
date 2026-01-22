import unittest
from oslo_config import iniparser
def test_assignment_colon(self):
    lines = ['foo: bar']
    self.parser.parse(lines)
    self.assertEqual({'': {'foo': ['bar']}}, self.parser.values)