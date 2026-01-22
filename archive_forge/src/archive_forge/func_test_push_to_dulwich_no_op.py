import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
def test_push_to_dulwich_no_op(self):
    self._old_repo = self.import_repo('server_old.export')
    self._new_repo = self.import_repo('server_old.export')
    self.assertReposEqual(self._old_repo, self._new_repo)
    port = self._start_server(self._old_repo)
    run_git_or_fail(['push', self.url(port), *self.branch_args()], cwd=self._new_repo.path)
    self.assertReposEqual(self._old_repo, self._new_repo)