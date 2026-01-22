from __future__ import annotations
import contextlib
import os
import subprocess
import sys
import warnings
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, Protocol, Union, overload
import trio
from ._core import ClosedResourceError, TaskStatus
from ._highlevel_generic import StapledStream
from ._subprocess_platform import (
from ._sync import Lock
from ._util import NoPublicConstructor, final
def pidfd_open(fd: int, flags: int) -> int:
    result = _cdll_for_pidfd_open.syscall(__NR_pidfd_open, fd, flags)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result