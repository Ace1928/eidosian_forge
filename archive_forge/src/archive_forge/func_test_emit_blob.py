import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_emit_blob(self):
    b = Blob()
    b.data = b'fooBAR'
    self.fastexporter.emit_blob(b)
    self.assertEqual(b'blob\nmark :1\ndata 6\nfooBAR\n', self.stream.getvalue())