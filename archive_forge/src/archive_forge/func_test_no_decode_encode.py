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
def test_no_decode_encode(self):
    repo_dir = tempfile.mkdtemp()
    repo_dir_bytes = os.fsencode(repo_dir)
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        file = Blob.from_string(b'foo')
        tree = Tree()
        latin1_name = 'À'.encode('latin1')
        latin1_path = os.path.join(repo_dir_bytes, latin1_name)
        utf8_name = 'À'.encode()
        utf8_path = os.path.join(repo_dir_bytes, utf8_name)
        tree[latin1_name] = (stat.S_IFREG | 420, file.id)
        tree[utf8_name] = (stat.S_IFREG | 420, file.id)
        repo.object_store.add_objects([(o, None) for o in [file, tree]])
        try:
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        except OSError as e:
            if e.errno == 92 and sys.platform == 'darwin':
                self.skipTest('can not write filename %r' % e.filename)
            else:
                raise
        except UnicodeDecodeError:
            self.skipTest('can not implicitly convert as utf8')
        index = repo.open_index()
        self.assertIn(latin1_name, index)
        self.assertIn(utf8_name, index)
        self.assertTrue(os.path.exists(latin1_path))
        self.assertTrue(os.path.exists(utf8_path))