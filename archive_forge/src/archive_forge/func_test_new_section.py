import unittest
from oslo_config import iniparser
def test_new_section(self):
    lines = ['[foo]']
    self.parser.parse(lines)
    self.assertEqual('foo', self.parser.section)