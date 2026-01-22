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
def test_pypy_issue(self):
    chunks = [b'tree 03207ccf58880a748188836155ceed72f03d65d6\nparent 408fbab530fd4abe49249a636a10f10f44d07a21\nauthor Victor Stinner <victor.stinner@gmail.com> 1421355207 +0100\ncommitter Victor Stinner <victor.stinner@gmail.com> 1421355207 +0100\n\nBackout changeset 3a06020af8cf\n\nStreamWriter: close() now clears the reference to the transport\n\nStreamWriter now raises an exception if it is closed: write(), writelines(),\nwrite_eof(), can_write_eof(), get_extra_info(), drain().\n']
    delta = [b'\xcd\x03\xad\x03]tree ff3c181a393d5a7270cddc01ea863818a8621ca8\nparent 20a103cc90135494162e819f98d0edfc1f1fba6b\x91]7\x0510738\x91\x99@\x0b10738 +0100\x93\x04\x01\xc9']
    res = apply_delta(chunks, delta)
    expected = [b'tree ff3c181a393d5a7270cddc01ea863818a8621ca8\nparent 20a103cc90135494162e819f98d0edfc1f1fba6b', b'\nauthor Victor Stinner <victor.stinner@gmail.com> 14213', b'10738', b' +0100\ncommitter Victor Stinner <victor.stinner@gmail.com> 14213', b'10738 +0100', b'\n\nStreamWriter: close() now clears the reference to the transport\n\nStreamWriter now raises an exception if it is closed: write(), writelines(),\nwrite_eof(), can_write_eof(), get_extra_info(), drain().\n']
    self.assertEqual(b''.join(expected), b''.join(res))