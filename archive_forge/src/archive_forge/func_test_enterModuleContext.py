import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterModuleContext(self):
    cleanups = []
    unittest.addModuleCleanup(cleanups.append, 'cleanup1')
    cm = TestCM(cleanups, 42)
    self.assertEqual(unittest.enterModuleContext(cm), 42)
    unittest.addModuleCleanup(cleanups.append, 'cleanup2')
    unittest.case.doModuleCleanups()
    self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])