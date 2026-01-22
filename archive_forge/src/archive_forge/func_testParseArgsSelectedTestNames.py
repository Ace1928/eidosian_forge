import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testParseArgsSelectedTestNames(self):
    program = self.program
    argv = ['progname', '-k', 'foo', '-k', 'bar', '-k', '*pat*']
    program.createTests = lambda: None
    program.parseArgs(argv)
    self.assertEqual(program.testNamePatterns, ['*foo*', '*bar*', '*pat*'])