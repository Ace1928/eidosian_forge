import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
def test_send_pack_new_ref_only(self):
    self.rin.write(b'0063310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00report-status delete-refs ofs-delta\n0000000eunpack ok\n0019ok refs/heads/blah12\n0000')
    self.rin.seek(0)

    def update_refs(refs):
        return {b'refs/heads/blah12': b'310ca9477129b8586fa2afc779c1f57cf64bba6c', b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'}

    def generate_pack_data(have, want, ofs_delta=False, progress=None):
        return (0, [])
    f = BytesIO()
    write_pack_objects(f.write, [])
    self.client.send_pack('/', update_refs, generate_pack_data)
    self.assertEqual(self.rout.getvalue(), b'008b0000000000000000000000000000000000000000 310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/blah12\x00delete-refs ofs-delta report-status0000' + f.getvalue())