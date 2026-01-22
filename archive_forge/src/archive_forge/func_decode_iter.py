from __future__ import annotations
import datetime
import itertools
import os
import re
import struct
import sys
import uuid
from codecs import utf_8_decode as _utf_8_decode
from codecs import utf_8_encode as _utf_8_encode
from collections import abc as _abc
from typing import (
from bson.binary import (
from bson.code import Code
from bson.codec_options import (
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.errors import InvalidBSON, InvalidDocument, InvalidStringData
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def decode_iter(data: bytes, codec_options: Optional[CodecOptions[_DocumentType]]=None) -> Union[Iterator[dict[str, Any]], Iterator[_DocumentType]]:
    """Decode BSON data to multiple documents as a generator.

    Works similarly to the decode_all function, but yields one document at a
    time.

    `data` must be a string of concatenated, valid, BSON-encoded
    documents.

    :Parameters:
      - `data`: BSON data
      - `codec_options` (optional): An instance of
        :class:`~bson.codec_options.CodecOptions`.

    .. versionchanged:: 3.0
       Replaced `as_class`, `tz_aware`, and `uuid_subtype` options with
       `codec_options`.

    .. versionadded:: 2.8
    """
    opts = codec_options or DEFAULT_CODEC_OPTIONS
    if not isinstance(opts, CodecOptions):
        raise _CODEC_OPTIONS_TYPE_ERROR
    position = 0
    end = len(data) - 1
    while position < end:
        obj_size = _UNPACK_INT_FROM(data, position)[0]
        elements = data[position:position + obj_size]
        position += obj_size
        yield _bson_to_dict(elements, opts)