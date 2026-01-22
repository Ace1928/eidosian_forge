from __future__ import annotations
import typing as t
import zlib
from ._json import _CompactJSON
from .encoding import base64_decode
from .encoding import base64_encode
from .exc import BadPayload
from .serializer import _PDataSerializer
from .serializer import Serializer
from .timed import TimedSerializer
def dump_payload(self, obj: t.Any) -> bytes:
    json = super().dump_payload(obj)
    is_compressed = False
    compressed = zlib.compress(json)
    if len(compressed) < len(json) - 1:
        json = compressed
        is_compressed = True
    base64d = base64_encode(json)
    if is_compressed:
        base64d = b'.' + base64d
    return base64d