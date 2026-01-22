from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_with_merges(self):
    committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
    c = commands.CommitCommand(b'refs/heads/master', b'ddd', None, committer, b'release v1.0', b':aaa', [b':bbb', b':ccc'], None)
    self.assertEqual(b'commit refs/heads/master\nmark :ddd\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nmerge :bbb\nmerge :ccc', bytes(c))