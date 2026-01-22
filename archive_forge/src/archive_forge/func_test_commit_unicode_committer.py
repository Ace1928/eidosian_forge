from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_unicode_committer(self):
    name = u'Ľórém Ípšúm'
    commit_utf8 = utf8_bytes_string(u'commit refs/heads/master\nmark :bbb\ncommitter %s <test@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa' % (name,))
    committer = (name, b'test@example.com', 1234567890, -6 * 3600)
    c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, None)
    self.assertEqual(commit_utf8, bytes(c))