import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_leading_commas(self):
    """Test multiline strings with leading commas"""
    tm = ip.magics_manager.registry['TerminalMagics']
    s = 'a = """\n,1,2,3\n"""'
    ip.user_ns.pop('foo', None)
    tm.store_or_execute(s, 'foo')
    self.assertIn('foo', ip.user_ns)