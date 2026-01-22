from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def enable_default_logging():
    """Configure default logging: messages to stderr and debug to brz.log

    This should only be called once per process.

    Non-command-line programs embedding breezy do not need to call this.  They
    can instead either pass a file to _push_log_file, or act directly on
    logging.getLogger("brz").

    Output can be redirected away by calling _push_log_file.

    :return: A memento from push_log_file for restoring the log state.
    """
    start_time = osutils.format_local_date(_brz_log_start_time, timezone='local')
    brz_log_file = _open_brz_log()
    if brz_log_file is not None:
        brz_log_file.write(start_time.encode('utf-8') + b'\n')
    memento = push_log_file(brz_log_file, '[%(process)5d] %(asctime)s.%(msecs)03d %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    logging.getLogger('brz').addHandler(stderr_handler)
    return memento