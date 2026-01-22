import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_echo(self):
    """Also test self.paste echoing, by temporarily faking the writer"""
    w = StringIO()
    old_write = sys.stdout.write
    sys.stdout.write = w.write
    code = '\n        a = 100\n        b = 200'
    try:
        self.paste(code, '')
        out = w.getvalue()
    finally:
        sys.stdout.write = old_write
    self.assertEqual(ip.user_ns['a'], 100)
    self.assertEqual(ip.user_ns['b'], 200)
    assert out == code + '\n## -- End pasted text --\n'