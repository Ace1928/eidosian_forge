import os
import shutil
import sys
import tempfile
import zlib
from hashlib import sha1
from io import BytesIO
from typing import Set
from dulwich.tests import TestCase
from ..errors import ApplyDeltaError, ChecksumMismatch
from ..file import GitFile
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, hex_to_sha, sha_to_hex
from ..pack import (
from .utils import build_pack, make_object
def test_get_object_at(self):
    """Tests random access for non-delta objects."""
    with self.get_pack(pack1_sha) as p:
        obj = p[a_sha]
        self.assertEqual(obj.type_name, b'blob')
        self.assertEqual(obj.sha().hexdigest().encode('ascii'), a_sha)
        obj = p[tree_sha]
        self.assertEqual(obj.type_name, b'tree')
        self.assertEqual(obj.sha().hexdigest().encode('ascii'), tree_sha)
        obj = p[commit_sha]
        self.assertEqual(obj.type_name, b'commit')
        self.assertEqual(obj.sha().hexdigest().encode('ascii'), commit_sha)