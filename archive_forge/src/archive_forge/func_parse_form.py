from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
def parse_form(headers, input_stream, on_field, on_file, chunk_size=1048576, **kwargs):
    """This function is useful if you just want to parse a request body,
    without too much work.  Pass it a dictionary-like object of the request's
    headers, and a file-like object for the input stream, along with two
    callbacks that will get called whenever a field or file is parsed.

    :param headers: A dictionary-like object of HTTP headers.  The only
                    required header is Content-Type.

    :param input_stream: A file-like object that represents the request body.
                         The read() method must return bytestrings.

    :param on_field: Callback to call with each parsed field.

    :param on_file: Callback to call with each parsed file.

    :param chunk_size: The maximum size to read from the input stream and write
                       to the parser at one time.  Defaults to 1 MiB.
    """
    parser = create_form_parser(headers, on_field, on_file)
    content_length = headers.get('Content-Length')
    if content_length is not None:
        content_length = int(content_length)
    else:
        content_length = float('inf')
    bytes_read = 0
    while True:
        max_readable = min(content_length - bytes_read, 1048576)
        buff = input_stream.read(max_readable)
        parser.write(buff)
        bytes_read += len(buff)
        if len(buff) != max_readable or bytes_read == content_length:
            break
    parser.finalize()