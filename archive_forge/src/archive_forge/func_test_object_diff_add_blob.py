from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_add_blob(self):
    f = BytesIO()
    store = MemoryObjectStore()
    b2 = Blob.from_string(b'new\nsame\n')
    store.add_object(b2)
    write_object_diff(f, store, (None, None, None), (b'bar.txt', 420, b2.id))
    self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'new file mode 644', b'index 0000000..a116b51', b'--- /dev/null', b'+++ b/bar.txt', b'@@ -0,0 +1,2 @@', b'+new', b'+same'], f.getvalue().splitlines())