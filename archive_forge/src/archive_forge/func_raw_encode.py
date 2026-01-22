from __future__ import annotations
import codecs
import os
import pickle
import sys
from collections import namedtuple
from contextlib import contextmanager
from io import BytesIO
from .exceptions import (ContentDisallowed, DecodeError, EncodeError,
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str, str_to_bytes
def raw_encode(data):
    """Special case serializer."""
    content_type = 'application/data'
    payload = data
    if isinstance(payload, str):
        content_encoding = 'utf-8'
        with _reraise_errors(EncodeError, exclude=()):
            payload = payload.encode(content_encoding)
    else:
        content_encoding = 'binary'
    return (content_type, content_encoding, payload)