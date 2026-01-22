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
def encode_multipart_data(data: RequestData, files: RequestFiles, boundary: bytes | None) -> tuple[dict[str, str], MultipartStream]:
    multipart = MultipartStream(data=data, files=files, boundary=boundary)
    headers = multipart.get_headers()
    return (headers, multipart)