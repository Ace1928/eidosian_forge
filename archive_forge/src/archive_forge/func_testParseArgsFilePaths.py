import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testParseArgsFilePaths(self):
    program = self.program
    argv = ['progname', 'foo/bar/baz.py', 'green\\red.py']
    self._patch_isfile(argv)
    program.createTests = lambda: None
    program.parseArgs(argv)
    expected = ['foo.bar.baz', 'green.red']
    self.assertEqual(program.testNames, expected)