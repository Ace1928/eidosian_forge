from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def communicate_with_process(process: subprocess.Popen, stdin: t.Optional[bytes], stdout: bool, stderr: bool, capture: bool, output_stream: OutputStream) -> tuple[bytes, bytes]:
    """Communicate with the specified process, handling stdin/stdout/stderr as requested."""
    threads: list[WrappedThread] = []
    reader: t.Type[ReaderThread]
    if capture:
        reader = CaptureThread
    else:
        reader = OutputThread
    if stdin is not None:
        threads.append(WriterThread(process.stdin, stdin))
    if stdout:
        stdout_reader = reader(process.stdout, output_stream.get_buffer(sys.stdout.buffer))
        threads.append(stdout_reader)
    else:
        stdout_reader = None
    if stderr:
        stderr_reader = reader(process.stderr, output_stream.get_buffer(sys.stderr.buffer))
        threads.append(stderr_reader)
    else:
        stderr_reader = None
    for thread in threads:
        thread.start()
    for thread in threads:
        try:
            thread.wait_for_result()
        except Exception as ex:
            display.error(str(ex))
    if isinstance(stdout_reader, ReaderThread):
        stdout_bytes = b''.join(stdout_reader.lines)
    else:
        stdout_bytes = b''
    if isinstance(stderr_reader, ReaderThread):
        stderr_bytes = b''.join(stderr_reader.lines)
    else:
        stderr_bytes = b''
    process.wait()
    return (stdout_bytes, stderr_bytes)