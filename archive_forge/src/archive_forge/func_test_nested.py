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
def test_nested(self):
    blob = Blob()
    blob.data = b'foo'
    self.store.add_object(blob)
    blobs = [(b'bla/bar', blob.id, stat.S_IFREG)]
    rootid = commit_tree(self.store, blobs)
    self.assertEqual(rootid, b'd92b959b216ad0d044671981196781b3258fa537')
    dirid = self.store[rootid][b'bla'][1]
    self.assertEqual(dirid, b'c1a1deb9788150829579a8b4efa6311e7b638650')
    self.assertEqual((stat.S_IFDIR, dirid), self.store[rootid][b'bla'])
    self.assertEqual((stat.S_IFREG, blob.id), self.store[dirid][b'bar'])
    self.assertEqual({rootid, dirid, blob.id}, set(self.store._data.keys()))