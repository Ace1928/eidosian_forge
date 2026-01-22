from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_bin_blob(self):
    f = BytesIO()
    b1 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b')
    b2 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3')
    store = MemoryObjectStore()
    store.add_objects([(b1, None), (b2, None)])
    write_object_diff(f, store, (b'foo.png', 420, b1.id), (b'bar.png', 420, b2.id))
    self.assertEqual([b'diff --git a/foo.png b/bar.png', b'index f73e47d..06364b7 644', b'Binary files a/foo.png and b/bar.png differ'], f.getvalue().splitlines())