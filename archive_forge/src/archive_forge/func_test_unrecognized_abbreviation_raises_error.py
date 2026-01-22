from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_unrecognized_abbreviation_raises_error(self):
    """If the abbreviation is unrecognized, then raise an error."""
    branch = self.make_branch('some_branch')
    self.assertRaises(bugtracker.UnknownBugTrackerAbbreviation, bugtracker.get_bug_url, 'xxx', branch, '1234')
    self.assertEqual([('get', 'xxx', branch)], self.tracker_type.log)