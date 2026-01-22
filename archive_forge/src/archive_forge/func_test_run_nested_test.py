import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_run_nested_test(self):
    ordering = []

    class InnerTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('inner setup')
            cls.addClassCleanup(ordering.append, 'inner cleanup')

        def test(self):
            ordering.append('inner test')

    class OuterTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('outer setup')
            cls.addClassCleanup(ordering.append, 'outer cleanup')

        def test(self):
            ordering.append('start outer test')
            runTests(InnerTest)
            ordering.append('end outer test')
    runTests(OuterTest)
    self.assertEqual(ordering, ['outer setup', 'start outer test', 'inner setup', 'inner test', 'inner cleanup', 'end outer test', 'outer cleanup'])