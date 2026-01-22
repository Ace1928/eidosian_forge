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
def test_mixed_chain(self):
    f = BytesIO()
    entries = build_pack(f, [(Blob.type_num, b'blob'), (REF_DELTA, (2, b'blob2')), (OFS_DELTA, (0, b'blob1')), (OFS_DELTA, (1, b'blob3')), (OFS_DELTA, (0, b'bob'))])
    self.assertEntriesMatch([0, 4, 2, 1, 3], entries, self.make_pack_iter(f))