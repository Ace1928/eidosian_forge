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
def test_commit_config_identity_from_env(self):
    self.overrideEnv('GIT_COMMITTER_NAME', 'joe')
    self.overrideEnv('GIT_COMMITTER_EMAIL', 'joe@example.com')
    r = self._repo
    c = r.get_config()
    c.set((b'user',), b'name', b'Jelmer')
    c.set((b'user',), b'email', b'jelmer@apache.org')
    c.write_to_path()
    commit_sha = r.do_commit(b'message')
    self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].author)
    self.assertEqual(b'joe <joe@example.com>', r[commit_sha].committer)