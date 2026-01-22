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
def test_long_chain(self):
    n = 100
    objects_spec = [(Blob.type_num, b'blob')]
    for i in range(n):
        objects_spec.append((OFS_DELTA, (i, b'blob' + str(i).encode('ascii'))))
    f = BytesIO()
    entries = build_pack(f, objects_spec)
    self.assertEntriesMatch(range(n + 1), entries, self.make_pack_iter(f))