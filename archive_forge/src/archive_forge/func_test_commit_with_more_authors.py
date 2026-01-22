from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_with_more_authors(self):
    author = (b'Sue Wong', b'sue@example.com', 1234565432, -6 * 3600)
    committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
    more_authors = [(b'Al Smith', b'al@example.com', 1234565432, -6 * 3600), (b'Bill Jones', b'bill@example.com', 1234565432, -6 * 3600)]
    c = commands.CommitCommand(b'refs/heads/master', b'bbb', author, committer, b'release v1.0', b':aaa', None, None, more_authors=more_authors)
    self.assertEqual(b'commit refs/heads/master\nmark :bbb\nauthor Sue Wong <sue@example.com> 1234565432 -0600\nauthor Al Smith <al@example.com> 1234565432 -0600\nauthor Bill Jones <bill@example.com> 1234565432 -0600\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))