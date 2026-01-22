from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_check_bug_id_doesnt_accept_non_integers(self):
    """A UniqueIntegerBugTracker rejects non-integers as bug IDs."""
    tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
    self.assertRaises(bugtracker.MalformedBugIdentifier, tracker.check_bug_id, 'red')