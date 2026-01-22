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
def test_clone_empty(self):
    """Test clone() doesn't crash if HEAD points to a non-existing ref.

        This simulates cloning server-side bare repository either when it is
        still empty or if user renames master branch and pushes private repo
        to the server.
        Non-bare repo HEAD always points to an existing ref.
        """
    r = self.open_repo('empty.git')
    tmp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    r.clone(tmp_dir, mkdir=False, bare=True)