from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
def submit_write(lpOverlapped: _Overlapped) -> None:
    offset_fields = lpOverlapped.DUMMYUNIONNAME.DUMMYSTRUCTNAME
    offset_fields.Offset = file_offset & 4294967295
    offset_fields.OffsetHigh = file_offset >> 32
    _check(kernel32.WriteFile(_handle(handle), ffi.cast('LPCVOID', cbuf), len(cbuf), ffi.NULL, lpOverlapped))