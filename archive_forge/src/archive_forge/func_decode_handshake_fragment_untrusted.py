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
def decode_handshake_fragment_untrusted(payload: bytes) -> HandshakeFragment:
    try:
        msg_type, msg_len_bytes, msg_seq, frag_offset_bytes, frag_len_bytes = HANDSHAKE_MESSAGE_HEADER.unpack_from(payload)
    except struct.error as exc:
        raise BadPacket('bad handshake message header') from exc
    msg_len = int.from_bytes(msg_len_bytes, 'big')
    frag_offset = int.from_bytes(frag_offset_bytes, 'big')
    frag_len = int.from_bytes(frag_len_bytes, 'big')
    frag = payload[HANDSHAKE_MESSAGE_HEADER.size:]
    if len(frag) != frag_len:
        raise BadPacket("handshake fragment length doesn't match record length")
    return HandshakeFragment(msg_type, msg_len, msg_seq, frag_offset, frag_len, frag)