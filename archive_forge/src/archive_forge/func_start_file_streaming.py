from __future__ import annotations
import typing as t
from io import BytesIO
from urllib.parse import parse_qsl
from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream
def start_file_streaming(self, event: File, total_content_length: int | None) -> t.IO[bytes]:
    content_type = event.headers.get('content-type')
    try:
        content_length = _plain_int(event.headers['content-length'])
    except (KeyError, ValueError):
        content_length = 0
    container = self.stream_factory(total_content_length=total_content_length, filename=event.filename, content_type=content_type, content_length=content_length)
    return container