import atexit
import errno
import math
import numbers
import os
import platform as _platform
import signal as _signal
import sys
import warnings
from contextlib import contextmanager
from billiard.compat import close_open_fds, get_fdmax
from billiard.util import set_pdeathsig as _set_pdeathsig
from kombu.utils.compat import maybe_fileno
from kombu.utils.encoding import safe_str
from .exceptions import SecurityError, SecurityWarning, reraise
from .local import try_import
def set_mp_process_title(progname, info=None, hostname=None):
    """Set the :command:`ps` name from the current process name.

        Only works if :pypi:`setproctitle` is installed.
        """
    if hostname:
        progname = f'{progname}: {hostname}'
    name = current_process().name if current_process else 'MainProcess'
    return set_process_title(f'{progname}:{name}', info=info)