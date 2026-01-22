import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
    """
        Set the filesystem root, the working directory, and daemonize.

        @type chroot: C{str} or L{None}
        @param chroot: If not None, a path to use as the filesystem root (using
            L{os.chroot}).

        @type rundir: C{str}
        @param rundir: The path to set as the working directory.

        @type nodaemon: C{bool}
        @param nodaemon: A flag which, if set, indicates that daemonization
            should not be done.

        @type umask: C{int} or L{None}
        @param umask: The value to which to change the process umask.

        @type pidfile: C{str} or L{None}
        @param pidfile: If not L{None}, the path to a file into which to put
            the PID of this process.
        """
    daemon = not nodaemon
    if chroot is not None:
        os.chroot(chroot)
        if rundir == '.':
            rundir = '/'
    os.chdir(rundir)
    if daemon and umask is None:
        umask = 63
    if umask is not None:
        os.umask(umask)
    if daemon:
        from twisted.internet import reactor
        self.config['statusPipe'] = self.daemonize(reactor)
    if pidfile:
        with open(pidfile, 'wb') as f:
            f.write(b'%d' % (os.getpid(),))