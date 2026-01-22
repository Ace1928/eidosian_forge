import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testTearDownUsesSuper(self):

    class OtherBaseCase(unittest.TestCase):
        tearDownCalled = False

        def tearDown(self):
            self.tearDownCalled = True
            super(OtherBaseCase, self).setUp()

    class OurCase(testresources.ResourcedTestCase, OtherBaseCase):

        def runTest(self):
            pass
    ourCase = OurCase()
    ourCase.setUp()
    ourCase.tearDown()
    self.assertTrue(ourCase.tearDownCalled)