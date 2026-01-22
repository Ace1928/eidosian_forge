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
def test_create_disk_bare_mkdir(self):
    tmp_dir = tempfile.mkdtemp()
    target_dir = os.path.join(tmp_dir, 'target')
    self.addCleanup(shutil.rmtree, tmp_dir)
    repo = Repo.init_bare(target_dir, mkdir=True)
    self.assertEqual(target_dir, repo._controldir)
    self._check_repo_contents(repo, True)