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
def test_create_index_v2(self):
    with self.get_pack_data(pack1_sha) as p:
        filename = os.path.join(self.tempdir, 'v2test.idx')
        p.create_index_v2(filename)
        idx1 = load_pack_index(filename)
        idx2 = self.get_pack_index(pack1_sha)
        self.assertEqual(oct(os.stat(filename).st_mode), indexmode)
        self.assertEqual(idx1, idx2)