import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def test_discovery_from_dotted_path(self):
    loader = unittest.TestLoader()
    tests = [self]
    expectedPath = os.path.abspath(os.path.dirname(unittest.test.__file__))
    self.wasRun = False

    def _find_tests(start_dir, pattern):
        self.wasRun = True
        self.assertEqual(start_dir, expectedPath)
        return tests
    loader._find_tests = _find_tests
    suite = loader.discover('unittest.test')
    self.assertTrue(self.wasRun)
    self.assertEqual(suite._tests, tests)