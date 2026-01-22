import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_command_line_handling_do_discovery_calls_loader(self):
    program = TestableTestProgram()

    class Loader(object):
        args = []

        def discover(self, start_dir, pattern, top_level_dir):
            self.args.append((start_dir, pattern, top_level_dir))
            return 'tests'
    program._do_discovery(['-v'], Loader=Loader)
    self.assertEqual(program.verbosity, 2)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('.', 'test*.py', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['--verbose'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('.', 'test*.py', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery([], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('.', 'test*.py', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['fish'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('fish', 'test*.py', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['fish', 'eggs'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('fish', 'eggs', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['fish', 'eggs', 'ham'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('fish', 'eggs', 'ham')])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['-s', 'fish'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('fish', 'test*.py', None)])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['-t', 'fish'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('.', 'test*.py', 'fish')])
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['-p', 'fish'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('.', 'fish', None)])
    self.assertFalse(program.failfast)
    self.assertFalse(program.catchbreak)
    Loader.args = []
    program = TestableTestProgram()
    program._do_discovery(['-p', 'eggs', '-s', 'fish', '-v', '-f', '-c'], Loader=Loader)
    self.assertEqual(program.test, 'tests')
    self.assertEqual(Loader.args, [('fish', 'eggs', None)])
    self.assertEqual(program.verbosity, 2)
    self.assertTrue(program.failfast)
    self.assertTrue(program.catchbreak)