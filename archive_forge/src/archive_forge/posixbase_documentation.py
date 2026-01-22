import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (

        Checks if the file descriptor is currently being observed for write
        readiness.

        @param fd: The file descriptor being checked.
        @type fd: L{twisted.internet.abstract.FileDescriptor}
        @return: C{True} if the file descriptor is being observed for write
            readiness, C{False} otherwise.
        @rtype: C{bool}
        