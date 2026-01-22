import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testSubTestMixed(self):
    with self.subTest('success', a=1):
        pass
    with self.subTest('skip', b=2):
        self.skipTest('skip')
    with self.subTest('fail', c=3):
        self.fail('fail')
    with self.subTest('error', d=4):
        raise Exception('error')