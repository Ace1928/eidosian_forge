from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING
import trio
from .. import _core, _subprocess
from .._abc import ReceiveStream, SendStream  # noqa: TCH001
def create_pipe_to_child_stdin():
    rh, wh = windows_pipe(overlapped=(False, True))
    return (PipeSendStream(wh), msvcrt.open_osfhandle(rh, os.O_RDONLY))