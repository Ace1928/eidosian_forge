import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_debug_executes_classCleanUp(self):
    ordering = []
    blowUp = False

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering, blowUp=blowUp)

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    suite.debug()
    self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_good'])
    ordering = []
    blowUp = True
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    with self.assertRaises(CustomError) as cm:
        suite.debug()
    self.assertEqual(str(cm.exception), 'CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_exc'])