import inspect
import os
import platform
import socket
import urllib.parse as urllib_parse
import warnings
from collections.abc import Sequence
from functools import reduce
from html import escape
from http import cookiejar as cookielib
from io import IOBase, StringIO as NativeStringIO, TextIOBase
from sys import intern
from types import FrameType, MethodType as _MethodType
from typing import Any, AnyStr, cast
from urllib.parse import quote as urlquote, unquote as urlunquote
from incremental import Version
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
def lazyByteSlice(object, offset=0, size=None):
    """
    Return a copy of the given bytes-like object.

    If an offset is given, the copy starts at that offset. If a size is
    given, the copy will only be of that length.

    @param object: C{bytes} to be copied.

    @param offset: C{int}, starting index of copy.

    @param size: Optional, if an C{int} is given limit the length of copy
        to this size.
    """
    view = memoryview(object)
    if size is None:
        return view[offset:]
    else:
        return view[offset:offset + size]