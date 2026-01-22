import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
@skipIf(not can_symlink(), 'Requires symlink support')
def test_symlink(self):
    repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        filed = Blob.from_string(b'file d')
        filee = Blob.from_string(b'd')
        tree = Tree()
        tree[b'c/d'] = (stat.S_IFREG | 420, filed.id)
        tree[b'c/e'] = (stat.S_IFLNK, filee.id)
        repo.object_store.add_objects([(o, None) for o in [filed, filee, tree]])
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        index = repo.open_index()
        epath = os.path.join(repo.path, 'c', 'e')
        self.assertTrue(os.path.exists(epath))
        self.assertReasonableIndexEntry(index[b'c/e'], stat.S_IFLNK, 0 if sys.platform == 'win32' else 1, filee.id)
        self.assertFileContents(epath, 'd', symlink=True)