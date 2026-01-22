import contextlib
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
import yaml
from oslo_utils import excutils
@contextlib.contextmanager
def remove_path_on_error(path, remove=delete_if_exists):
    """Protect code that wants to operate on PATH atomically.
    Any exception will cause PATH to be removed.

    :param path: File to work with
    :param remove: Optional function to remove passed path
    """
    try:
        yield
    except Exception:
        with excutils.save_and_reraise_exception():
            remove(path)