import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_comparingPrereleases(self) -> None:
    """
        The value specified as the prerelease is used in version comparisons.
        """
    va = Version('whatever', 1, 0, 0, prerelease=1)
    vb = Version('whatever', 1, 0, 0, prerelease=2)
    self.assertTrue(va < vb)
    self.assertTrue(vb > va)
    self.assertTrue(va <= vb)
    self.assertTrue(vb >= va)
    self.assertTrue(va != vb)
    self.assertTrue(vb == Version('whatever', 1, 0, 0, prerelease=2))
    self.assertTrue(va == va)