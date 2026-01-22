from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import (
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import (
from ._utils import peek_filelike_length, primitive_value_to_str
def encode_content(content: str | bytes | Iterable[bytes] | AsyncIterable[bytes]) -> tuple[dict[str, str], SyncByteStream | AsyncByteStream]:
    if isinstance(content, (bytes, str)):
        body = content.encode('utf-8') if isinstance(content, str) else content
        content_length = len(body)
        headers = {'Content-Length': str(content_length)} if body else {}
        return (headers, ByteStream(body))
    elif isinstance(content, Iterable) and (not isinstance(content, dict)):
        content_length_or_none = peek_filelike_length(content)
        if content_length_or_none is None:
            headers = {'Transfer-Encoding': 'chunked'}
        else:
            headers = {'Content-Length': str(content_length_or_none)}
        return (headers, IteratorByteStream(content))
    elif isinstance(content, AsyncIterable):
        headers = {'Transfer-Encoding': 'chunked'}
        return (headers, AsyncIteratorByteStream(content))
    raise TypeError(f"Unexpected type for 'content', {type(content)!r}")