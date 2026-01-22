import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_add_alternate_path(self):
    store = DiskObjectStore(self.store_dir)
    self.assertEqual([], list(store._read_alternate_paths()))
    store.add_alternate_path('/foo/path')
    self.assertEqual(['/foo/path'], list(store._read_alternate_paths()))
    store.add_alternate_path('/bar/path')
    self.assertEqual(['/foo/path', '/bar/path'], list(store._read_alternate_paths()))