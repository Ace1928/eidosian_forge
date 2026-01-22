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
def test_commit_fail_ref(self):
    r = self._repo

    def set_if_equals(name, old_ref, new_ref, **kwargs):
        return False
    r.refs.set_if_equals = set_if_equals

    def add_if_new(name, new_ref, **kwargs):
        self.fail('Unexpected call to add_if_new')
    r.refs.add_if_new = add_if_new
    old_shas = set(r.object_store)
    self.assertRaises(errors.CommitError, r.do_commit, b'failed commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    new_shas = set(r.object_store) - old_shas
    self.assertEqual(1, len(new_shas))
    new_commit = r[new_shas.pop()]
    self.assertEqual(r[self._root_commit].tree, new_commit.tree)
    self.assertEqual(b'failed commit', new_commit.message)