import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_full_mismatched_brackets(self):
    try:
        list(interpolate('${{1:2}'))
    except TemplateSyntaxError as e:
        pass
    else:
        self.fail('Expected TemplateSyntaxError')