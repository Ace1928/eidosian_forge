import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_is_reserved_id(self):
    self.assertEqual(True, revision.is_reserved_id(NULL_REVISION))
    self.assertEqual(True, revision.is_reserved_id(revision.CURRENT_REVISION))
    self.assertEqual(True, revision.is_reserved_id(b'arch:'))
    self.assertEqual(False, revision.is_reserved_id(b'null'))
    self.assertEqual(False, revision.is_reserved_id(b'arch:a@example.com/c--b--v--r'))
    self.assertEqual(False, revision.is_reserved_id(None))