import unittest
from oslo_config import iniparser
def test_section_assignment(self):
    lines = ['[test]', 'foo = bar']
    self.parser.parse(lines)
    self.assertEqual({'test': {'foo': ['bar']}}, self.parser.values)