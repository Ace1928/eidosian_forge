import os
import shutil
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode
from jupyter_server import _tz as tz
from .checkpoints import (
from .fileio import AsyncFileManagerMixin, FileManagerMixin
def checkpoint_path(self, checkpoint_id, path):
    """find the path to a checkpoint"""
    path = path.strip('/')
    parent, name = ('/' + path).rsplit('/', 1)
    parent = parent.strip('/')
    basename, ext = os.path.splitext(name)
    filename = f'{basename}-{checkpoint_id}{ext}'
    os_path = self._get_os_path(path=parent)
    cp_dir = os.path.join(os_path, self.checkpoint_dir)
    with self.perm_to_403():
        ensure_dir_exists(cp_dir)
    cp_path = os.path.join(cp_dir, filename)
    return cp_path