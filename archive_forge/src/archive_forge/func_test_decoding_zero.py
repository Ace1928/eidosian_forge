from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_decoding_zero(self):
    self.assertEqual([], list(bugtracker.decode_bug_urls('')))