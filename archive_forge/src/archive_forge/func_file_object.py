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
@property
def file_object(self):
    """The file object that we're currently writing to.  Note that this
        will either be an instance of a :class:`io.BytesIO`, or a regular file
        object.
        """
    return self._fileobj