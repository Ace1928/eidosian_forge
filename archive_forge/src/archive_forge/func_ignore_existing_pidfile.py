import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def ignore_existing_pidfile():
    """Normally, daemonize() or daemonize_start() will terminate the program
    with a message if a locked pidfile already exists.  If this function is
    called, an existing pidfile will be replaced, with a warning."""
    global _overwrite_pidfile
    _overwrite_pidfile = True