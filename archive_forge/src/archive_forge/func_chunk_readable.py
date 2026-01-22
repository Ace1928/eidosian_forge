from __future__ import annotations
import io
import typing
from base64 import b64encode
from enum import Enum
from ..exceptions import UnrewindableBodyError
from .util import to_bytes
def chunk_readable() -> typing.Iterable[bytes]:
    nonlocal body, blocksize
    encode = isinstance(body, io.TextIOBase)
    while True:
        datablock = body.read(blocksize)
        if not datablock:
            break
        if encode:
            datablock = datablock.encode('iso-8859-1')
        yield datablock