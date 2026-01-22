import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_dobuleescaped(self):
    parts = list(interpolate('$$${bla}'))
    self.assertEqual(2, len(parts))
    self.assertEqual(TEXT, parts[0][0])
    self.assertEqual('$', parts[0][1])
    self.assertEqual(EXPR, parts[1][0])
    self.assertEqual('bla', parts[1][1].source)