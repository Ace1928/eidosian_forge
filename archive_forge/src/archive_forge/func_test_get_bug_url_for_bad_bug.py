from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_get_bug_url_for_bad_bug(self):
    """When given a bug identifier that is invalid for Trac, get_bug_url
        should raise an error.
        """
    self.assertRaises(bugtracker.MalformedBugIdentifier, self.tracker.get_bug_url, 'bad')