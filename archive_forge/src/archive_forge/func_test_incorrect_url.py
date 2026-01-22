from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_incorrect_url(self):
    err = bugtracker.InvalidBugTrackerURL('foo', 'http://bug.example.com/')
    self.assertEqual('The URL for bug tracker "foo" doesn\'t contain {id}: http://bug.example.com/', str(err))