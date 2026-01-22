import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def test_ExitAsDefault(self):
    stream = BufferedWriter()
    self.assertRaises(SystemExit, unittest.main, argv=['foobar'], testRunner=unittest.TextTestRunner(stream=stream), testLoader=self.FooBarLoader())
    out = stream.getvalue()
    self.assertIn('\nFAIL: testFail ', out)
    self.assertIn('\nERROR: testError ', out)
    self.assertIn('\nUNEXPECTED SUCCESS: testUnexpectedSuccess ', out)
    expected = '\n\nFAILED (failures=1, errors=1, skipped=1, expected failures=1, unexpected successes=1)\n'
    self.assertTrue(out.endswith(expected))