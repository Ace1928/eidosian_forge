import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_get_summary(self):
    r = revision.Revision('1')
    r.message = 'a'
    self.assertEqual('a', r.get_summary())
    r.message = 'a\nb'
    self.assertEqual('a', r.get_summary())
    r.message = '\na\nb'
    self.assertEqual('a', r.get_summary())
    r.message = None
    self.assertEqual('', r.get_summary())