import unittest
from oslo_config import iniparser
def test_assignment_multiline(self):
    lines = ['foo = bar0', '  bar1']
    self.parser.parse(lines)
    self.assertEqual({'': {'foo': ['bar0', 'bar1']}}, self.parser.values)