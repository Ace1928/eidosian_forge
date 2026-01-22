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
def set_pidfile(name):
    """Sets up a following call to daemonize() to create a pidfile named
    'name'.  If 'name' begins with '/', then it is treated as an absolute path.
    Otherwise, it is taken relative to ovs.util.RUNDIR, which is
    $(prefix)/var/run by default.

    If 'name' is null, then ovs.util.PROGRAM_NAME followed by ".pid" is
    used."""
    global _pidfile
    _pidfile = make_pidfile_name(name)