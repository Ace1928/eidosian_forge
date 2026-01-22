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
def test_reset_index_symlink_disabled(self):
    tmp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    o = Repo.init(os.path.join(tmp_dir, 's'), mkdir=True)
    o.close()
    os.symlink('foo', os.path.join(tmp_dir, 's', 'bar'))
    o.stage('bar')
    o.do_commit(b'add symlink')
    t = o.clone(os.path.join(tmp_dir, 't'), symlinks=False)
    with open(os.path.join(tmp_dir, 't', 'bar')) as f:
        self.assertEqual('foo', f.read())
    t.close()