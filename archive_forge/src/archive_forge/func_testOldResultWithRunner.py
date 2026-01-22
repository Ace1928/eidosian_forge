import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testOldResultWithRunner(self):

    class Test(unittest.TestCase):

        def testFoo(self):
            pass
    runner = unittest.TextTestRunner(resultclass=OldResult, stream=io.StringIO())
    runner.run(Test('testFoo'))