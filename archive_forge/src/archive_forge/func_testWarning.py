import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testWarning(self):
    """Test the warnings argument"""

    class FakeTP(unittest.TestProgram):

        def parseArgs(self, *args, **kw):
            pass

        def runTests(self, *args, **kw):
            pass
    warnoptions = sys.warnoptions[:]
    try:
        sys.warnoptions[:] = []
        self.assertEqual(FakeTP().warnings, 'default')
        self.assertEqual(FakeTP(warnings='ignore').warnings, 'ignore')
        sys.warnoptions[:] = ['somevalue']
        self.assertEqual(FakeTP().warnings, None)
        self.assertEqual(FakeTP(warnings='ignore').warnings, 'ignore')
    finally:
        sys.warnoptions[:] = warnoptions