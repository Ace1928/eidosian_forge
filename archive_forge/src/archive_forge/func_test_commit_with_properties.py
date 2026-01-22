from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_with_properties(self):
    committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
    properties = {u'greeting': u'hello', u'planet': u'world'}
    c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, None, properties=properties)
    self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nproperty greeting 5 hello\nproperty planet 5 world', bytes(c))