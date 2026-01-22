import atexit
import os
import signal
import sys
import ovs.vlog
def unlink_file_now(file):
    """Like fatal_signal_remove_file_to_unlink(), but also unlinks 'file'.
    Returns 0 if successful, otherwise a positive errno value."""
    error = _unlink(file)
    if error:
        vlog.warn('could not unlink "%s" (%s)' % (file, os.strerror(error)))
    remove_file_to_unlink(file)
    return error