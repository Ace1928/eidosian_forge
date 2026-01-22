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
def records_untrusted(packet: bytes) -> Iterator[Record]:
    i = 0
    while i < len(packet):
        try:
            ct, version, epoch_seqno, payload_len = RECORD_HEADER.unpack_from(packet, i)
        except struct.error as exc:
            raise BadPacket('invalid record header') from exc
        i += RECORD_HEADER.size
        payload = packet[i:i + payload_len]
        if len(payload) != payload_len:
            raise BadPacket('short record')
        i += payload_len
        yield Record(ct, version, epoch_seqno, payload)