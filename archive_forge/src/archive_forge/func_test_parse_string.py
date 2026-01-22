import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_parse_string(self):
    self.assertEqual(bp.chars_no_quotecurly.parseString('x')[0], 'x')
    self.assertEqual(bp.chars_no_quotecurly.parseString('a string')[0], 'a string')
    self.assertEqual(bp.chars_no_quotecurly.parseString('a "string')[0], 'a ')
    self.assertEqual(bp.chars_no_curly.parseString('x')[0], 'x')
    self.assertEqual(bp.chars_no_curly.parseString('a string')[0], 'a string')
    self.assertEqual(bp.chars_no_curly.parseString('a {string')[0], 'a ')
    self.assertEqual(bp.chars_no_curly.parseString('a }string')[0], 'a ')
    for obj in (bp.curly_string, bp.string, bp.field_value):
        self.assertEqual(obj.parseString('{}').asList(), [])
        self.assertEqual(obj.parseString('{a "string}')[0], 'a "string')
        self.assertEqual(obj.parseString('{a {nested} string}').asList(), ['a ', ['nested'], ' string'])
        self.assertEqual(obj.parseString('{a {double {nested}} string}').asList(), ['a ', ['double ', ['nested']], ' string'])
    for obj in (bp.quoted_string, bp.string, bp.field_value):
        self.assertEqual(obj.parseString('""').asList(), [])
        self.assertEqual(obj.parseString('"a string"')[0], 'a string')
        self.assertEqual(obj.parseString('"a {nested} string"').asList(), ['a ', ['nested'], ' string'])
        self.assertEqual(obj.parseString('"a {double {nested}} string"').asList(), ['a ', ['double ', ['nested']], ' string'])
    self.assertEqual(bp.string.parseString('someascii')[0], Macro('someascii'))
    self.assertRaises(ParseException, bp.string.parseString, '%#= validstring')
    self.assertEqual(bp.string.parseString('1994')[0], '1994')