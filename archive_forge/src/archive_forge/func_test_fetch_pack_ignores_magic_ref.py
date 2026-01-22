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
def test_fetch_pack_ignores_magic_ref(self):
    self.rin.write(b'00000000000000000000000000000000000000000000 capabilities^{}\x00 multi_ack thin-pack side-band side-band-64k ofs-delta shallow no-progress include-tag\n0000')
    self.rin.seek(0)

    def check_heads(heads, **kwargs):
        self.assertEqual({}, heads)
        return []
    ret = self.client.fetch_pack(b'bla', check_heads, None, None, None)
    self.assertEqual({}, ret.refs)
    self.assertEqual({}, ret.symrefs)
    self.assertEqual(self.rout.getvalue(), b'0000')