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
def enable_insecure_serializers(choices=NOTSET):
    """Enable serializers that are considered to be unsafe.

    Note:
    ----
        Will enable ``pickle``, ``yaml`` and ``msgpack`` by default, but you
        can also specify a list of serializers (by name or content type)
        to enable.
    """
    choices = ['pickle', 'yaml', 'msgpack'] if choices is NOTSET else choices
    if choices is not None:
        for choice in choices:
            try:
                registry.enable(choice)
            except KeyError:
                pass