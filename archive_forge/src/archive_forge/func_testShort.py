import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def testShort(self) -> None:
    self.assertEqual(Version('dummy', 1, 2, 3).short(), '1.2.3')