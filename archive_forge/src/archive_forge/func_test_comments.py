import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_comments(self):
    res = bp.comment.parseString('@Comment{about something}')
    self.assertEqual(res.asList(), ['comment', '{about something}'])
    self.assertEqual(bp.comment.parseString('@COMMENT{about something').asList(), ['comment', '{about something'])
    self.assertEqual(bp.comment.parseString('@comment(about something').asList(), ['comment', '(about something'])
    self.assertEqual(bp.comment.parseString('@COMment about something').asList(), ['comment', ' about something'])
    self.assertRaises(ParseException, bp.comment.parseString, '@commentabout something')
    self.assertRaises(ParseException, bp.comment.parseString, '@comment+about something')
    self.assertRaises(ParseException, bp.comment.parseString, '@comment"about something')