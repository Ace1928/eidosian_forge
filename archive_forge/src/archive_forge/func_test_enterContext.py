import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterContext(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    test = TestableTest('testNothing')
    cleanups = []
    test.addCleanup(cleanups.append, 'cleanup1')
    cm = TestCM(cleanups, 42)
    self.assertEqual(test.enterContext(cm), 42)
    test.addCleanup(cleanups.append, 'cleanup2')
    self.assertTrue(test.doCleanups())
    self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])