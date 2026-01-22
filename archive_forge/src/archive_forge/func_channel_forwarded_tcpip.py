import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def channel_forwarded_tcpip(self, windowSize, maxPacket, data):
    log.msg(f'FTCP {data!r}')
    remoteHP, origHP = forwarding.unpackOpen_forwarded_tcpip(data)
    log.msg(self.remoteForwards)
    log.msg(remoteHP)
    if remoteHP[1] in self.remoteForwards:
        connectHP = self.remoteForwards[remoteHP[1]]
        log.msg(f'connect forwarding {connectHP}')
        return SSHConnectForwardingChannel(connectHP, remoteWindow=windowSize, remoteMaxPacket=maxPacket, conn=self)
    else:
        raise ConchError(connection.OPEN_CONNECT_FAILED, "don't know about that port")