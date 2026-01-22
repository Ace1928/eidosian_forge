from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_kind_change(self):
    f = BytesIO()
    b1 = Blob.from_string(b'new\nsame\n')
    store = MemoryObjectStore()
    store.add_object(b1)
    write_object_diff(f, store, (b'bar.txt', 420, b1.id), (b'bar.txt', 57344, b'06d0bdd9e2e20377b3180e4986b14c8549b393e4'))
    self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'old file mode 644', b'new file mode 160000', b'index a116b51..06d0bdd 160000', b'--- a/bar.txt', b'+++ b/bar.txt', b'@@ -1,2 +1 @@', b'-new', b'-same', b'+Subproject commit 06d0bdd9e2e20377b3180e4986b14c8549b393e4'], f.getvalue().splitlines())