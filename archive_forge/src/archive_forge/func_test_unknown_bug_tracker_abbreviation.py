from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_unknown_bug_tracker_abbreviation(self):
    """Test the formatting of UnknownBugTrackerAbbreviation."""
    branch = self.make_branch('some_branch')
    error = bugtracker.UnknownBugTrackerAbbreviation('xxx', branch)
    self.assertEqual('Cannot find registered bug tracker called xxx on %s' % branch, str(error))