from __future__ import annotations
from collections.abc import Iterable
from contextlib import contextmanager
import logging
import sys
import textwrap
from typing import Iterator
from typing import Optional
from typing import TextIO
from typing import Union
import warnings
from sqlalchemy.engine import url
from . import sqla_compat
def write_outstream(stream: TextIO, *text: Union[str, bytes], quiet: bool=False) -> None:
    if quiet:
        return
    encoding = getattr(stream, 'encoding', 'ascii') or 'ascii'
    for t in text:
        if not isinstance(t, bytes):
            t = t.encode(encoding, 'replace')
        t = t.decode(encoding)
        try:
            stream.write(t)
        except OSError:
            break