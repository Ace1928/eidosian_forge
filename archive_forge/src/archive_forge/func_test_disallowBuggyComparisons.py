import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_disallowBuggyComparisons(self) -> None:
    """
        The package names of the Version objects need to be the same,
        """
    self.assertRaises(IncomparableVersions, operator.eq, Version('dummy', 1, 0, 0), Version('dumym', 1, 0, 0))