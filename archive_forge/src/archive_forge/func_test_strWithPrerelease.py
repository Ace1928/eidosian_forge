import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_strWithPrerelease(self) -> None:
    """
        Calling C{str} on a version with a prerelease includes the prerelease.
        """
    self.assertEqual(str(Version('dummy', 1, 0, 0, prerelease=1)), '[dummy, version 1.0.0.rc1]')