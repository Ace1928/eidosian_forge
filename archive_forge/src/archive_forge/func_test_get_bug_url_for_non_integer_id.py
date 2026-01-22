from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_get_bug_url_for_non_integer_id(self):
    self.tracker.check_bug_id('ABC-1234')