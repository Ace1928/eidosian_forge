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
def test_commit_follows(self):
    r = self._repo
    r.refs.set_symbolic_ref(b'HEAD', b'refs/heads/bla')
    commit_sha = r.do_commit(b'commit with strange character', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'HEAD')
    self.assertEqual(commit_sha, r[b'refs/heads/bla'].id)