import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
def content_disposition_filename(params: Mapping[str, str], name: str='filename') -> Optional[str]:
    name_suf = '%s*' % name
    if not params:
        return None
    elif name_suf in params:
        return params[name_suf]
    elif name in params:
        return params[name]
    else:
        parts = []
        fnparams = sorted(((key, value) for key, value in params.items() if key.startswith(name_suf)))
        for num, (key, value) in enumerate(fnparams):
            _, tail = key.split('*', 1)
            if tail.endswith('*'):
                tail = tail[:-1]
            if tail == str(num):
                parts.append(value)
            else:
                break
        if not parts:
            return None
        value = ''.join(parts)
        if "'" in value:
            encoding, _, value = value.split("'", 2)
            encoding = encoding or 'utf-8'
            return unquote(value, encoding, 'strict')
        return value