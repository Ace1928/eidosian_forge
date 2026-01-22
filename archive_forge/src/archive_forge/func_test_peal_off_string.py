from bpython.test import unittest
from bpython.curtsiesfrontend import parse
from curtsies.fmtfuncs import yellow, cyan, green, bold
def test_peal_off_string(self):
    self.assertEqual(parse.peel_off_string('\x01RI\x03]\x04asdf'), ({'bg': 'I', 'string': ']', 'fg': 'R', 'colormarker': '\x01RI', 'bold': ''}, 'asdf'))