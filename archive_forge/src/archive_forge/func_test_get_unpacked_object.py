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
def test_get_unpacked_object(self):
    self.maxDiff = None
    with self.make_pack(False) as p:
        expected = UnpackedObject(7, delta_base=b'\x19\x10(\x15f=#\xf8\xb7ZG\xe7\xa0\x19e\xdc\xdc\x96F\x8c', decomp_chunks=[b'\x03\x07\x90\x03\x041234'])
        expected.offset = 12
        got = p.get_unpacked_object(self.blobs[b'foo1234'].id)
        self.assertEqual(expected, got)
    with self.make_pack(True) as p:
        expected = UnpackedObject(7, delta_base=b'\x19\x10(\x15f=#\xf8\xb7ZG\xe7\xa0\x19e\xdc\xdc\x96F\x8c', decomp_chunks=[b'\x03\x07\x90\x03\x041234'])
        expected.offset = 12
        got = p.get_unpacked_object(self.blobs[b'foo1234'].id)
        self.assertEqual(expected, got)