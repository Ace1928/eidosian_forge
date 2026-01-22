import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_debug_module_executes_cleanUp(self):
    ordering = []
    blowUp = False

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup, ordering, blowUp=blowUp)

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    TestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    suite.debug()
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good'])
    self.assertEqual(unittest.case._module_cleanups, [])
    ordering = []
    blowUp = True
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    with self.assertRaises(CustomError) as cm:
        suite.debug()
    self.assertEqual(str(cm.exception), 'CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])
    self.assertEqual(unittest.case._module_cleanups, [])