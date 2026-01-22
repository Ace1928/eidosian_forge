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
def path_to_invalid(path):
    """Name of invalid file after a failed atomic write and subsequent read."""
    dirname, basename = os.path.split(path)
    return os.path.join(dirname, basename + '.invalid')