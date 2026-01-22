from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
def sendBox(self, box):
    """
        Send a amp.Box to my peer.

        Note: transport.write is never called outside of this method.

        @param box: an AmpBox.

        @raise ProtocolSwitched: if the protocol has previously been switched.

        @raise ConnectionLost: if the connection has previously been lost.
        """
    if self._locked:
        raise ProtocolSwitched('This connection has switched: no AMP traffic allowed.')
    if self.transport is None:
        raise ConnectionLost()
    if self._startingTLSBuffer is not None:
        self._startingTLSBuffer.append(box)
    else:
        self.transport.write(box.serialize())