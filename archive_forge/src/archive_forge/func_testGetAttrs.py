import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@skipUnless(sys.platform.startswith('win'), 'requires Windows')
def testGetAttrs(self):
    term = WinTerm()
    term._fore = 0
    term._back = 0
    term._style = 0
    self.assertEqual(term.get_attrs(), 0)
    term._fore = WinColor.YELLOW
    self.assertEqual(term.get_attrs(), WinColor.YELLOW)
    term._back = WinColor.MAGENTA
    self.assertEqual(term.get_attrs(), WinColor.YELLOW + WinColor.MAGENTA * 16)
    term._style = WinStyle.BRIGHT
    self.assertEqual(term.get_attrs(), WinColor.YELLOW + WinColor.MAGENTA * 16 + WinStyle.BRIGHT)