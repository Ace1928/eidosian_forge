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
def test_decompress_truncated(self):
    read = BytesIO(self.comp[:10]).read
    self.assertRaises(zlib.error, read_zlib_chunks, read, self.unpacked)
    read = BytesIO(self.comp).read
    self.assertRaises(zlib.error, read_zlib_chunks, read, self.unpacked)