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
def test_get_raw(self):
    with self.make_pack(False) as p:
        self.assertRaises(KeyError, p.get_raw, self.blobs[b'foo1234'].id)
    with self.make_pack(True) as p:
        self.assertEqual((3, b'foo1234'), p.get_raw(self.blobs[b'foo1234'].id))