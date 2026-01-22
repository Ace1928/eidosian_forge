import unittest
from oslo_config import iniparser
def test_assignment_multline_empty(self):
    lines = ['foo = bar0', '', '  bar1']
    self.assertRaises(iniparser.ParseError, self.parser.parse, lines)