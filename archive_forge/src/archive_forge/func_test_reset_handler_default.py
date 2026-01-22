import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_reset_handler_default(self):
    from fastimport import commands
    [c1, c2] = build_commit_graph(self.repo.object_store, [[1], [2]])
    cmd = commands.ResetCommand(b'refs/heads/foo', None)
    self.processor.reset_handler(cmd)
    self.assertEqual(ZERO_SHA, self.repo.get_refs()[b'refs/heads/foo'])