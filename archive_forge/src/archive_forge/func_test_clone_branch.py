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
def test_clone_branch(self):
    r = self.open_repo('a.git')
    r.refs[b'refs/heads/mybranch'] = b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a'
    tmp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    with r.clone(tmp_dir, mkdir=False, branch=b'mybranch') as t:
        chain, sha = t.refs.follow(b'HEAD')
        self.assertEqual(chain[-1], b'refs/heads/mybranch')
        self.assertEqual(sha, b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a')
        self.assertEqual(t.refs[b'refs/remotes/origin/HEAD'], b'a90fa2d900a17e99b433217e988c4eb4a2e9a097')