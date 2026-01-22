import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
@skipIf(sys.platform == 'win32', 'fails on Windows')
def test_fetch_ignores_missing_refs(self):
    r = self.open_repo('a.git')
    missing = b'1234566789123456789123567891234657373833'
    r.refs[b'refs/heads/blah'] = missing
    tmp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    t = Repo.init(tmp_dir)
    self.addCleanup(t.close)
    r.fetch(t)
    self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
    self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
    self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
    self.assertIn(b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', t)
    self.assertIn(b'b0931cadc54336e78a1d980420e3268903b57a50', t)
    self.assertNotIn(missing, t)