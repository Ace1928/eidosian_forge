import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def get_null_fd():
    """Returns a readable and writable fd for /dev/null, if successful,
    otherwise a negative errno value.  The caller must not close the returned
    fd (because the same fd will be handed out to subsequent callers)."""
    global null_fd
    if null_fd < 0:
        try:
            null_fd = os.open(os.devnull, os.O_RDWR)
        except OSError as e:
            vlog.err('could not open %s: %s' % (os.devnull, os.strerror(e.errno)))
            return -e.errno
    return null_fd