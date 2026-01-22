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
def test_compute_file_sha_short_file(self):
    f = BytesIO(b'abcd1234wxyz')
    self.assertRaises(AssertionError, compute_file_sha, f, end_ofs=-20)
    self.assertRaises(AssertionError, compute_file_sha, f, end_ofs=20)
    self.assertRaises(AssertionError, compute_file_sha, f, start_ofs=10, end_ofs=-12)