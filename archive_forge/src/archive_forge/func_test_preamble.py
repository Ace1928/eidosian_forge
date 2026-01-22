import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_preamble(self):
    res = bp.preamble.parseString('@preamble{"about something"}')
    self.assertEqual(res.asList(), ['preamble', 'about something'])
    self.assertEqual(bp.preamble.parseString('@PREamble{{about something}}').asList(), ['preamble', 'about something'])
    self.assertEqual(bp.preamble.parseString('@PREamble{\n            {about something}\n        }').asList(), ['preamble', 'about something'])