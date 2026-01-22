import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterModuleContext_arg_errors(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        unittest.enterModuleContext(LacksEnterAndExit())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        unittest.enterModuleContext(LacksEnter())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        unittest.enterModuleContext(LacksExit())
    self.assertEqual(unittest.case._module_cleanups, [])