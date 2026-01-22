import logging
import os
import tempfile
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import contextmanager
import warnings
import platformdirs
from packaging.version import Version
def make_local_storage(path, env=None):
    """
    Create the local cache directory and make sure it's writable.

    Parameters
    ----------
    path : str or PathLike
        The path to the local data storage folder.
    env : str or None
        An environment variable that can be used to overwrite *path*. Only used
        in the error message in case the folder is not writable.
    """
    path = str(path)
    if not os.path.exists(path):
        action = 'create'
    else:
        action = 'write to'
    try:
        if action == 'create':
            os.makedirs(path, exist_ok=True)
        else:
            with tempfile.NamedTemporaryFile(dir=path):
                pass
    except PermissionError as error:
        message = [str(error), f"| Pooch could not {action} data cache folder '{path}'.", 'Will not be able to download data files.']
        if env is not None:
            message.append(f"Use environment variable '{env}' to specify a different location.")
        raise PermissionError(' '.join(message)) from error