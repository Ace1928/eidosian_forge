import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def test_NonExit(self):
    stream = BufferedWriter()
    program = unittest.main(exit=False, argv=['foobar'], testRunner=unittest.TextTestRunner(stream=stream), testLoader=self.FooBarLoader())
    self.assertTrue(hasattr(program, 'result'))
    out = stream.getvalue()
    self.assertIn('\nFAIL: testFail ', out)
    self.assertIn('\nERROR: testError ', out)
    self.assertIn('\nUNEXPECTED SUCCESS: testUnexpectedSuccess ', out)
    expected = '\n\nFAILED (failures=1, errors=1, skipped=1, expected failures=1, unexpected successes=1)\n'
    self.assertTrue(out.endswith(expected))