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
def encode_handshake_fragment(hsf: HandshakeFragment) -> bytes:
    hs_header = HANDSHAKE_MESSAGE_HEADER.pack(hsf.msg_type, hsf.msg_len.to_bytes(3, 'big'), hsf.msg_seq, hsf.frag_offset.to_bytes(3, 'big'), hsf.frag_len.to_bytes(3, 'big'))
    return hs_header + hsf.frag