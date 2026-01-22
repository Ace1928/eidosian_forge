import sys
from unittest import TestCase, main
from ..ansi import Back, Fore, Style
from ..ansitowin32 import AnsiToWin32
def testForeAttributes(self):
    self.assertEqual(Fore.BLACK, '\x1b[30m')
    self.assertEqual(Fore.RED, '\x1b[31m')
    self.assertEqual(Fore.GREEN, '\x1b[32m')
    self.assertEqual(Fore.YELLOW, '\x1b[33m')
    self.assertEqual(Fore.BLUE, '\x1b[34m')
    self.assertEqual(Fore.MAGENTA, '\x1b[35m')
    self.assertEqual(Fore.CYAN, '\x1b[36m')
    self.assertEqual(Fore.WHITE, '\x1b[37m')
    self.assertEqual(Fore.RESET, '\x1b[39m')
    self.assertEqual(Fore.LIGHTBLACK_EX, '\x1b[90m')
    self.assertEqual(Fore.LIGHTRED_EX, '\x1b[91m')
    self.assertEqual(Fore.LIGHTGREEN_EX, '\x1b[92m')
    self.assertEqual(Fore.LIGHTYELLOW_EX, '\x1b[93m')
    self.assertEqual(Fore.LIGHTBLUE_EX, '\x1b[94m')
    self.assertEqual(Fore.LIGHTMAGENTA_EX, '\x1b[95m')
    self.assertEqual(Fore.LIGHTCYAN_EX, '\x1b[96m')
    self.assertEqual(Fore.LIGHTWHITE_EX, '\x1b[97m')