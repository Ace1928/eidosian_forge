from __future__ import annotations
import errno
import hashlib
import os
import shutil
from base64 import decodebytes, encodebytes
from contextlib import contextmanager
from functools import partial
import nbformat
from anyio.to_thread import run_sync
from tornado.web import HTTPError
from traitlets import Bool, Enum
from traitlets.config import Configurable
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.utils import ApiPath, to_api_path, to_os_path
@contextmanager
def perm_to_403(self, os_path=''):
    """context manager for turning permission errors into 403."""
    try:
        yield
    except OSError as e:
        if e.errno in {errno.EPERM, errno.EACCES}:
            if not os_path:
                os_path = e.filename or 'unknown file'
            path = to_api_path(os_path, root=self.root_dir)
            raise HTTPError(403, 'Permission denied: %s' % path) from e
        else:
            raise