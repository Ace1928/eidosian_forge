import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testMainInstallsHandler(self):
    failfast = object()
    test = object()
    verbosity = object()
    result = object()
    default_handler = signal.getsignal(signal.SIGINT)

    class FakeRunner(object):
        initArgs = []
        runArgs = []

        def __init__(self, *args, **kwargs):
            self.initArgs.append((args, kwargs))

        def run(self, test):
            self.runArgs.append(test)
            return result

    class Program(unittest.TestProgram):

        def __init__(self, catchbreak):
            self.exit = False
            self.verbosity = verbosity
            self.failfast = failfast
            self.catchbreak = catchbreak
            self.tb_locals = False
            self.testRunner = FakeRunner
            self.test = test
            self.result = None
    p = Program(False)
    p.runTests()
    self.assertEqual(FakeRunner.initArgs, [((), {'buffer': None, 'verbosity': verbosity, 'failfast': failfast, 'tb_locals': False, 'warnings': None})])
    self.assertEqual(FakeRunner.runArgs, [test])
    self.assertEqual(p.result, result)
    self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
    FakeRunner.initArgs = []
    FakeRunner.runArgs = []
    p = Program(True)
    p.runTests()
    self.assertEqual(FakeRunner.initArgs, [((), {'buffer': None, 'verbosity': verbosity, 'failfast': failfast, 'tb_locals': False, 'warnings': None})])
    self.assertEqual(FakeRunner.runArgs, [test])
    self.assertEqual(p.result, result)
    self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)