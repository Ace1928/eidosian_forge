import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_trailing_question(self):
    """Test pasting sources with trailing question marks"""
    tm = ip.magics_manager.registry['TerminalMagics']
    s = "def funcfoo():\n   if True: #am i true?\n       return 'fooresult'\n"
    ip.user_ns.pop('funcfoo', None)
    self.paste(s)
    self.assertEqual(ip.user_ns['funcfoo'](), 'fooresult')