import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def patchUserDatabase(patch, user, uid, group, gid):
    """
    Patch L{pwd.getpwnam} so that it behaves as though only one user exists
    and patch L{grp.getgrnam} so that it behaves as though only one group
    exists.

    @param patch: A function like L{TestCase.patch} which will be used to
        install the fake implementations.

    @type user: C{str}
    @param user: The name of the single user which will exist.

    @type uid: C{int}
    @param uid: The UID of the single user which will exist.

    @type group: C{str}
    @param group: The name of the single user which will exist.

    @type gid: C{int}
    @param gid: The GID of the single group which will exist.
    """
    pwent = pwd.getpwuid(os.getuid())
    grent = grp.getgrgid(os.getgid())
    database = UserDatabase()
    database.addUser(user, pwent.pw_passwd, uid, gid, pwent.pw_gecos, pwent.pw_dir, pwent.pw_shell)

    def getgrnam(name):
        result = list(grent)
        result[result.index(grent.gr_name)] = group
        result[result.index(grent.gr_gid)] = gid
        result = tuple(result)
        return {group: result}[name]
    patch(pwd, 'getpwnam', database.getpwnam)
    patch(grp, 'getgrnam', getgrnam)
    patch(pwd, 'getpwuid', database.getpwuid)