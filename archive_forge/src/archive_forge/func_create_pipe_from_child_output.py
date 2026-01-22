from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING
import trio
from .. import _core, _subprocess
from .._abc import ReceiveStream, SendStream  # noqa: TCH001
def create_pipe_from_child_output():
    rh, wh = windows_pipe(overlapped=(True, False))
    return (PipeReceiveStream(rh), msvcrt.open_osfhandle(wh, 0))