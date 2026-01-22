import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
@_pause_echo(echo_input)
def visible_input(prompt: t.Optional[str]=None) -> str:
    sys.stdout.write(prompt or '')
    val = text_input.readline().rstrip('\r\n')
    sys.stdout.write(f'{val}\n')
    sys.stdout.flush()
    return val