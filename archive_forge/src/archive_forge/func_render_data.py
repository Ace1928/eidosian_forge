from __future__ import annotations
import io
import os
import typing
from pathlib import Path
from ._types import (
from ._utils import (
def render_data(self) -> typing.Iterator[bytes]:
    if isinstance(self.file, (str, bytes)):
        yield to_bytes(self.file)
        return
    if hasattr(self.file, 'seek'):
        try:
            self.file.seek(0)
        except io.UnsupportedOperation:
            pass
    chunk = self.file.read(self.CHUNK_SIZE)
    while chunk:
        yield to_bytes(chunk)
        chunk = self.file.read(self.CHUNK_SIZE)