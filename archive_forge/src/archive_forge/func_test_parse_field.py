import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_parse_field(self):
    fv = bp.field_value
    self.assertEqual(fv.parseString('aname')[0], Macro('aname'))
    self.assertEqual(fv.parseString('ANAME')[0], Macro('aname'))
    self.assertEqual(fv.parseString('aname # "some string"').asList(), [Macro('aname'), 'some string'])
    self.assertEqual(fv.parseString('aname # {some {string}}').asList(), [Macro('aname'), 'some ', ['string']])
    self.assertEqual(fv.parseString('"a string" # 1994').asList(), ['a string', '1994'])
    self.assertEqual(fv.parseString('"a string" # 1994 # a_macro').asList(), ['a string', '1994', Macro('a_macro')])