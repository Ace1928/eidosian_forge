import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def request_shell(self, data):
    log.info('Getting shell')
    if not self.session:
        self.session = ISession(self.avatar)
    try:
        pp = SSHSessionProcessProtocol(self)
        self.session.openShell(pp)
    except Exception:
        log.failure('Error getting shell')
        return 0
    else:
        self.client = pp
        return 1