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
def str2time(s: Union[str, bytes, int, None]) -> Union[int, None]:
    """
    Parse a string description of an interval into an integer number of seconds.

    @param s: An interval definition constructed as an interval duration
        followed by an interval unit.  An interval duration is a base ten
        representation of an integer.  An interval unit is one of the following
        letters: S (seconds), M (minutes), H (hours), D (days), W (weeks), or Y
        (years).  For example: C{"3S"} indicates an interval of three seconds;
        C{"5D"} indicates an interval of five days.  Alternatively, C{s} may be
        any non-string and it will be returned unmodified.
    @type s: text string (L{bytes} or L{str}) for parsing; anything else
        for passthrough.

    @return: an L{int} giving the interval represented by the string C{s}, or
        whatever C{s} is if it is not a string.
    """
    if isinstance(s, bytes):
        return _str2time(s.decode('ascii'))
    if isinstance(s, str):
        return _str2time(s)
    return s