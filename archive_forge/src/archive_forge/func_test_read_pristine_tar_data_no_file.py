import stat
from base64 import standard_b64encode
from dulwich.objects import Blob, Tree
from dulwich.repo import MemoryRepo as GitMemoryRepo
from ...revision import Revision
from ...tests import TestCase
from ..pristine_tar import (get_pristine_tar_tree, read_git_pristine_tar_data,
def test_read_pristine_tar_data_no_file(self):
    r = GitMemoryRepo()
    t = Tree()
    b = Blob.from_string(b'README')
    r.object_store.add_object(b)
    t.add(b'README', stat.S_IFREG | 420, b.id)
    r.object_store.add_object(t)
    r.do_commit(b'Add README', tree=t.id, ref=b'refs/heads/pristine-tar')
    self.assertRaises(KeyError, read_git_pristine_tar_data, r, b'foo')