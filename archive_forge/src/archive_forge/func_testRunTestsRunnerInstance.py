import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testRunTestsRunnerInstance(self):
    program = self.program
    program.testRunner = FakeRunner()
    FakeRunner.initArgs = None
    program.runTests()
    self.assertIsNone(FakeRunner.initArgs)
    self.assertEqual(FakeRunner.test, 'test')
    self.assertIs(program.result, RESULT)