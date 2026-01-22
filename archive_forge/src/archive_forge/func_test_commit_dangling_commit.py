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
def test_commit_dangling_commit(self):
    r = self._repo
    old_shas = set(r.object_store)
    old_refs = r.get_refs()
    commit_sha = r.do_commit(b'commit with no ref', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=None)
    new_shas = set(r.object_store) - old_shas
    self.assertEqual(1, len(new_shas))
    new_commit = r[new_shas.pop()]
    self.assertEqual(r[self._root_commit].tree, new_commit.tree)
    self.assertEqual([], r[commit_sha].parents)
    self.assertEqual(old_refs, r.get_refs())