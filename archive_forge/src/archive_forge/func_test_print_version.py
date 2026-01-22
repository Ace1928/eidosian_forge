import sys
import unittest
from Cython.Utils import (
def test_print_version(self):
    orig_stderr = sys.stderr
    orig_stdout = sys.stdout
    stderr = sys.stderr = StringIO()
    stdout = sys.stdout = StringIO()
    try:
        print_version()
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    stdout = stdout.getvalue()
    stderr = stderr.getvalue()
    from .. import __version__ as version
    self.assertIn(version, stdout)
    if stderr:
        self.assertIn(version, stderr)