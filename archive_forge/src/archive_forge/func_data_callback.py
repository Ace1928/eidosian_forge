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
def data_callback(name, remaining=False):
    marked_index = self.marks.get(name)
    if marked_index is None:
        return
    if remaining:
        self.callback(name, data, marked_index, length)
        self.marks[name] = 0
    else:
        self.callback(name, data, marked_index, i)
        self.marks.pop(name, None)