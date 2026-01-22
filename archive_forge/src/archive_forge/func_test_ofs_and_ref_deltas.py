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
def test_ofs_and_ref_deltas(self):
    f = BytesIO()
    entries = build_pack(f, [(REF_DELTA, (1, b'blob1')), (Blob.type_num, b'blob'), (OFS_DELTA, (1, b'blob2'))])
    self.assertEntriesMatch([1, 0, 2], entries, self.make_pack_iter(f))