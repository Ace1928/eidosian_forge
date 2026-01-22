import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_full_nested_brackets(self):
    parts = list(interpolate('${{1:2}}'))
    self.assertEqual(1, len(parts))
    self.assertEqual(EXPR, parts[0][0])
    self.assertEqual('{1:2}', parts[0][1].source)