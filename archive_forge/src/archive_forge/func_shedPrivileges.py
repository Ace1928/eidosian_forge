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
def shedPrivileges(self, euid, uid, gid):
    """
        Change the UID and GID or the EUID and EGID of this process.

        @type euid: C{bool}
        @param euid: A flag which, if set, indicates that only the I{effective}
            UID and GID should be set.

        @type uid: C{int} or L{None}
        @param uid: If not L{None}, the UID to which to switch.

        @type gid: C{int} or L{None}
        @param gid: If not L{None}, the GID to which to switch.
        """
    if uid is not None or gid is not None:
        extra = euid and 'e' or ''
        desc = f'{extra}uid/{extra}gid {uid}/{gid}'
        try:
            switchUID(uid, gid, euid)
        except OSError as e:
            log.msg('failed to set {}: {} (are you root?) -- exiting.'.format(desc, e))
            sys.exit(1)
        else:
            log.msg(f'set {desc}')