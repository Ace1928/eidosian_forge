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
def test_write_pack_object(self):
    f = BytesIO()
    f.write(b'header')
    offset = f.tell()
    crc32 = write_pack_object(f.write, Blob.type_num, b'blob')
    self.assertEqual(crc32, zlib.crc32(f.getvalue()[6:]) & 4294967295)
    f.write(b'x')
    f.seek(offset)
    unpacked, unused = unpack_object(f.read, compute_crc32=True)
    self.assertEqual(Blob.type_num, unpacked.pack_type_num)
    self.assertEqual(Blob.type_num, unpacked.obj_type_num)
    self.assertEqual([b'blob'], unpacked.decomp_chunks)
    self.assertEqual(crc32, unpacked.crc32)
    self.assertEqual(b'x', unused)