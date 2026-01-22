import io
import sys
import unittest
def test_skiptest_in_setupclass(self):

    class Test(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest('foo')

        def test_one(self):
            pass

        def test_two(self):
            pass
    result = self.runTests(Test)
    self.assertEqual(result.testsRun, 0)
    self.assertEqual(len(result.errors), 0)
    self.assertEqual(len(result.skipped), 1)
    skipped = result.skipped[0][0]
    self.assertEqual(str(skipped), 'setUpClass (%s.%s)' % (__name__, Test.__qualname__))