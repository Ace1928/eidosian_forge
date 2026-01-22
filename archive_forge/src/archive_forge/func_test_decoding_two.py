from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_decoding_two(self):
    self.assertEqual([('http://example.com/bugs/1', 'fixed'), ('http://example.com/bugs/2', 'related')], list(bugtracker.decode_bug_urls('http://example.com/bugs/1 fixed\nhttp://example.com/bugs/2 related')))