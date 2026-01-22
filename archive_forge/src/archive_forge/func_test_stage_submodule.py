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
def test_stage_submodule(self):
    r = self._repo
    s = Repo.init(os.path.join(r.path, 'sub'), mkdir=True)
    s.do_commit(b'message')
    r.stage(['sub'])
    self.assertEqual([b'a', b'sub'], list(r.open_index()))