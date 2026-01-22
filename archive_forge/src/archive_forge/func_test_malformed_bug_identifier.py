from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_malformed_bug_identifier(self):
    """Test the formatting of MalformedBugIdentifier."""
    error = bugtracker.MalformedBugIdentifier('bogus', 'reason for bogosity')
    self.assertEqual('Did not understand bug identifier bogus: reason for bogosity. See "brz help bugs" for more information on this feature.', str(error))