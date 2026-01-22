from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
def read_volley() -> list[_AnyHandshakeMessage]:
    volley_bytes = _read_loop(self._ssl.bio_read)
    new_volley_messages = decode_volley_trusted(volley_bytes)
    if new_volley_messages and volley_messages and isinstance(new_volley_messages[0], HandshakeMessage) and isinstance(volley_messages[0], HandshakeMessage) and (new_volley_messages[0].msg_seq == volley_messages[0].msg_seq):
        return []
    else:
        return new_volley_messages