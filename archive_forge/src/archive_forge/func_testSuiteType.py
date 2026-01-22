import testtools
from testresources import TestLoader, OptimisingTestSuite
from testresources.tests import TestUtil
def testSuiteType(self):
    loader = TestLoader()
    suite = loader.loadTestsFromName(__name__)
    self.assertIsInstance(suite, OptimisingTestSuite)