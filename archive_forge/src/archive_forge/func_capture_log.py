import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
@contextmanager
def capture_log(level=logging.DEBUG):
    """
    Create a context manager for reading from the logs.

    Yields
    ------
    log_file : StringIO
        a file-like object to which the logs were written
    """
    log_file = io.StringIO()
    handler = logging.StreamHandler(log_file)
    handler.setLevel(level)
    get_logger().addHandler(handler)
    yield log_file
    get_logger().removeHandler(handler)