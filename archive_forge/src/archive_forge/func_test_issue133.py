import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
def test_issue133(self):
    input = textwrap.dedent('            def spam(a, (b, c)):\n            pass\n            \x08\n            spam(1')
    return self.run_bpython(input).addCallback(self.check_no_traceback)