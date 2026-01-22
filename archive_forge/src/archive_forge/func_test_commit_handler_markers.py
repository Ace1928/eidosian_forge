import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_commit_handler_markers(self):
    from fastimport import commands
    [c1, c2, c3] = build_commit_graph(self.repo.object_store, [[1], [2], [3]])
    self.processor.markers[b'10'] = c1.id
    self.processor.markers[b'42'] = c2.id
    self.processor.markers[b'98'] = c3.id
    cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', b':10', [b':42', b':98'], [])
    self.processor.commit_handler(cmd)
    commit = self.repo[self.processor.last_commit]
    self.assertEqual(c1.id, commit.parents[0])
    self.assertEqual(c2.id, commit.parents[1])
    self.assertEqual(c3.id, commit.parents[2])