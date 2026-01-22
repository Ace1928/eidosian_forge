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
def wrapProcessProtocol(inst):
    if isinstance(inst, protocol.Protocol):
        return _ProtocolWrapper(inst)
    else:
        return inst