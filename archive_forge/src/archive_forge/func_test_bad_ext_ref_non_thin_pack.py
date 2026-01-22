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
def test_bad_ext_ref_non_thin_pack(self):
    blob, = self.store_blobs([b'blob'])
    f = BytesIO()
    build_pack(f, [(REF_DELTA, (blob.id, b'blob1'))], store=self.store)
    pack_iter = self.make_pack_iter(f, thin=False)
    try:
        list(pack_iter._walk_all_chains())
        self.fail()
    except UnresolvedDeltas as e:
        self.assertEqual([blob.id], e.shas)