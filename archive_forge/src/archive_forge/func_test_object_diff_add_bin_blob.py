from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_add_bin_blob(self):
    f = BytesIO()
    b2 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3')
    store = MemoryObjectStore()
    store.add_object(b2)
    write_object_diff(f, store, (None, None, None), (b'bar.png', 420, b2.id))
    self.assertEqual([b'diff --git a/bar.png b/bar.png', b'new file mode 644', b'index 0000000..06364b7', b'Binary files /dev/null and b/bar.png differ'], f.getvalue().splitlines())