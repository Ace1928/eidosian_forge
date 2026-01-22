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
def make_pidfile_name(name):
    """Returns the file name that would be used for a pidfile if 'name' were
    provided to set_pidfile()."""
    if name is None or name == '':
        return '%s/%s.pid' % (ovs.dirs.RUNDIR, ovs.util.PROGRAM_NAME)
    else:
        return ovs.util.abs_file_name(ovs.dirs.RUNDIR, name)