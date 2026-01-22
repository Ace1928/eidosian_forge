import os
import shutil
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode
from jupyter_server import _tz as tz
from .checkpoints import (
from .fileio import AsyncFileManagerMixin, FileManagerMixin
def checkpoint_model(self, checkpoint_id, os_path):
    """construct the info dict for a given checkpoint"""
    stats = os.stat(os_path)
    last_modified = tz.utcfromtimestamp(stats.st_mtime)
    info = {'id': checkpoint_id, 'last_modified': last_modified}
    return info