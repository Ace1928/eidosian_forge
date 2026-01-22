import io
import sys
import unittest
def test_class_not_setup_or_torndown_when_skipped(self):

    class Test(unittest.TestCase):
        classSetUp = False
        tornDown = False

        @classmethod
        def setUpClass(cls):
            Test.classSetUp = True

        @classmethod
        def tearDownClass(cls):
            Test.tornDown = True

        def test_one(self):
            pass
    Test = unittest.skip('hop')(Test)
    self.runTests(Test)
    self.assertFalse(Test.classSetUp)
    self.assertFalse(Test.tornDown)