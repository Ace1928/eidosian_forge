import io
import sys
import unittest
def test_skiptest_in_setupmodule(self):

    class Test(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass

    class Module(object):

        @staticmethod
        def setUpModule():
            raise unittest.SkipTest('foo')
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = self.runTests(Test)
    self.assertEqual(result.testsRun, 0)
    self.assertEqual(len(result.errors), 0)
    self.assertEqual(len(result.skipped), 1)
    skipped = result.skipped[0][0]
    self.assertEqual(str(skipped), 'setUpModule (Module)')