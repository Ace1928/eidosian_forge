from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def lookupRecordType(self, type):
    """
        Retrieve the L{IRecord} implementation for the given record type.

        @param type: A record type, such as C{A} or L{NS}.
        @type type: L{int}

        @return: An object which implements L{IRecord} or L{None} if none
            can be found for the given type.
        @rtype: C{Type[IRecord]}
        """
    return self._recordTypes.get(type, UnknownRecord)