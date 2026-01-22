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
def fromStringProto(self, inString, proto):
    """
        Take a unique identifier associated with a file descriptor which must
        have been received by now and use it to look up that descriptor in a
        dictionary where they are kept.

        @param inString: The base representation (as a byte string) of an
            ordinal indicating which file descriptor corresponds to this usage
            of this argument.
        @type inString: C{str}

        @param proto: The protocol used to receive this descriptor.  This
            protocol must be connected via a transport providing
            L{IUNIXTransport<twisted.internet.interfaces.IUNIXTransport>}.
        @type proto: L{BinaryBoxProtocol}

        @return: The file descriptor represented by C{inString}.
        @rtype: C{int}
        """
    return proto._getDescriptor(int(inString))