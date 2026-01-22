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
def test_lsremote_from_dulwich(self):
    self._repo = self.import_repo('server_old.export')
    port = self._start_server(self._repo)
    o = run_git_or_fail(['ls-remote', self.url(port)])
    self.assertEqual(len(o.split(b'\n')), 4)