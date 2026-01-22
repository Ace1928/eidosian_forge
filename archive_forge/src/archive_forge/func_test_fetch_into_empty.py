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
def test_fetch_into_empty(self):
    c = LocalGitClient()
    target = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, target)
    t = Repo.init_bare(target)
    self.addCleanup(t.close)
    s = open_repo('a.git')
    self.addCleanup(tear_down_repo, s)
    self.assertEqual(s.get_refs(), c.fetch(s.path, t).refs)