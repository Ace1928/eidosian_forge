import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
@contextlib.contextmanager
def modified_environ(update):
    """Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    Args:
        update: Dictionary of environment variables and values to add/update.
    """
    update = update or {}
    original_env = {k: os.environ.get(k) for k in update}
    try:
        os.environ.update(update)
        yield
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v