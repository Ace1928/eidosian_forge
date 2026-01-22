import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_short_starting_with_digit(self):
    parts = list(interpolate('$0bla'))
    self.assertEqual(1, len(parts))
    self.assertEqual(TEXT, parts[0][0])
    self.assertEqual('$0bla', parts[0][1])