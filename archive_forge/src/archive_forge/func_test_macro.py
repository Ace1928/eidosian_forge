import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_macro(self):
    res = bp.macro.parseString('@string{ANAME = "about something"}')
    self.assertEqual(res.asList(), ['string', 'aname', 'about something'])
    self.assertEqual(bp.macro.parseString('@string{aname = {about something}}').asList(), ['string', 'aname', 'about something'])