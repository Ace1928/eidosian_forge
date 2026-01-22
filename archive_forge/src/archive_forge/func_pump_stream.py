from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def pump_stream(cmdline: List[str], name: str, stream: Union[BinaryIO, TextIO], is_decode: bool, handler: Union[None, Callable[[Union[bytes, str]], None]]) -> None:
    try:
        for line in stream:
            if handler:
                if is_decode:
                    assert isinstance(line, bytes)
                    line_str = line.decode(defenc)
                    handler(line_str)
                else:
                    handler(line)
    except Exception as ex:
        _logger.error(f'Pumping {name!r} of cmd({remove_password_if_present(cmdline)}) failed due to: {ex!r}')
        if 'I/O operation on closed file' not in str(ex):
            raise CommandError([f'<{name}-pump>'] + remove_password_if_present(cmdline), ex) from ex
    finally:
        stream.close()