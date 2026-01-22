import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_short_containing_digit(self):
    parts = list(interpolate('$foo0'))
    self.assertEqual(1, len(parts))
    self.assertEqual(EXPR, parts[0][0])
    self.assertEqual('foo0', parts[0][1].source)