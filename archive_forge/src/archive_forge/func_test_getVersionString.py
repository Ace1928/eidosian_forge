import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_getVersionString(self) -> None:
    """
        L{getVersionString} returns a string with the package name and the
        short version number.
        """
    self.assertEqual('Twisted 8.0.0', getVersionString(Version('Twisted', 8, 0, 0)))