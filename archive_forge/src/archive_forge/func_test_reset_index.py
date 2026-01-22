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
def test_reset_index(self):
    r = self._repo
    with open(os.path.join(r.path, 'a'), 'wb') as f:
        f.write(b'changed')
    with open(os.path.join(r.path, 'b'), 'wb') as f:
        f.write(b'added')
    r.stage(['a', 'b'])
    status = list(porcelain.status(self._repo))
    self.assertEqual([{'add': [b'b'], 'delete': [], 'modify': [b'a']}, [], []], status)
    r.reset_index()
    status = list(porcelain.status(self._repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['b']], status)