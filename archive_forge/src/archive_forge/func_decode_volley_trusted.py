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
def decode_volley_trusted(volley: bytes) -> list[_AnyHandshakeMessage]:
    messages: list[_AnyHandshakeMessage] = []
    messages_by_seq = {}
    for record in records_untrusted(volley):
        if record.epoch_seqno & EPOCH_MASK:
            messages.append(OpaqueHandshakeMessage(record))
        elif record.content_type in (ContentType.change_cipher_spec, ContentType.alert):
            messages.append(PseudoHandshakeMessage(record.version, record.content_type, record.payload))
        else:
            assert record.content_type == ContentType.handshake
            fragment = decode_handshake_fragment_untrusted(record.payload)
            msg_type = HandshakeType(fragment.msg_type)
            if fragment.msg_seq not in messages_by_seq:
                msg = HandshakeMessage(record.version, msg_type, fragment.msg_seq, bytearray(fragment.msg_len))
                messages.append(msg)
                messages_by_seq[fragment.msg_seq] = msg
            else:
                msg = messages_by_seq[fragment.msg_seq]
            assert msg.msg_type == fragment.msg_type
            assert msg.msg_seq == fragment.msg_seq
            assert len(msg.body) == fragment.msg_len
            msg.body[fragment.frag_offset:fragment.frag_offset + fragment.frag_len] = fragment.frag
    return messages