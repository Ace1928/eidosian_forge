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
def test_pack_tuples(self):
    with self.get_pack(pack1_sha) as p:
        tuples = p.pack_tuples()
        expected = {(p[s], None) for s in [commit_sha, tree_sha, a_sha]}
        self.assertEqual(expected, set(list(tuples)))
        self.assertEqual(expected, set(list(tuples)))
        self.assertEqual(3, len(tuples))