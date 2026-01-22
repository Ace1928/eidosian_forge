import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_py_multi_r(self):
    """Now, test that self.paste -r works"""
    self.test_paste_py_multi()
    self.assertEqual(ip.user_ns.pop('x'), [1, 2, 3])
    self.assertEqual(ip.user_ns.pop('y'), [1, 4, 9])
    self.assertFalse('x' in ip.user_ns)
    ip.run_line_magic('paste', '-r')
    self.assertEqual(ip.user_ns['x'], [1, 2, 3])
    self.assertEqual(ip.user_ns['y'], [1, 4, 9])