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
def request_exec(self, data):
    if not self.session:
        self.session = ISession(self.avatar)
    f, data = common.getNS(data)
    log.info('Executing command "{f}"', f=f)
    try:
        pp = SSHSessionProcessProtocol(self)
        self.session.execCommand(pp, f)
    except Exception:
        log.failure('Error executing command "{f}"', f=f)
        return 0
    else:
        self.client = pp
        return 1