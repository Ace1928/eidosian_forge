import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testNoExit(self):
    result = object()
    test = object()

    class FakeRunner(object):

        def run(self, test):
            self.test = test
            return result
    runner = FakeRunner()
    oldParseArgs = unittest.TestProgram.parseArgs

    def restoreParseArgs():
        unittest.TestProgram.parseArgs = oldParseArgs
    unittest.TestProgram.parseArgs = lambda *args: None
    self.addCleanup(restoreParseArgs)

    def removeTest():
        del unittest.TestProgram.test
    unittest.TestProgram.test = test
    self.addCleanup(removeTest)
    program = unittest.TestProgram(testRunner=runner, exit=False, verbosity=2)
    self.assertEqual(program.result, result)
    self.assertEqual(runner.test, test)
    self.assertEqual(program.verbosity, 2)