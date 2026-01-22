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
def read_all_from_possibly_closed_stream(stream: Union[IO[bytes], None]) -> bytes:
    if stream:
        try:
            return stderr_b + force_bytes(stream.read())
        except (OSError, ValueError):
            return stderr_b or b''
    else:
        return stderr_b or b''