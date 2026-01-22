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
def remove_if_stale(self):
    """Remove the lock if the process isn't running.

        I.e. process does not respond to signal.
        """
    try:
        pid = self.read_pid()
    except ValueError:
        print('Broken pidfile found - Removing it.', file=sys.stderr)
        self.remove()
        return True
    if not pid:
        self.remove()
        return True
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH or exc.errno == errno.EPERM:
            print('Stale pidfile exists - Removing it.', file=sys.stderr)
            self.remove()
            return True
    except SystemError:
        print('Stale pidfile exists - Removing it.', file=sys.stderr)
        self.remove()
        return True
    return False