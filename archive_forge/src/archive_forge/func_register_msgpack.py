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
def register_msgpack():
    """Register msgpack serializer.

    See Also
    --------
        https://msgpack.org/.
    """
    pack = unpack = None
    try:
        import msgpack
        if msgpack.version >= (0, 4):
            from msgpack import packb, unpackb

            def pack(s):
                return packb(s, use_bin_type=True)

            def unpack(s):
                return unpackb(s, raw=False)
        else:

            def version_mismatch(*args, **kwargs):
                raise SerializerNotInstalled('msgpack requires msgpack-python >= 0.4.0')
            pack = unpack = version_mismatch
    except (ImportError, ValueError):

        def not_available(*args, **kwargs):
            raise SerializerNotInstalled('No decoder installed for msgpack. Please install the msgpack-python library')
        pack = unpack = not_available
    registry.register('msgpack', pack, unpack, content_type='application/x-msgpack', content_encoding='binary')