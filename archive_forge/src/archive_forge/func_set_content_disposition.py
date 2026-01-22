import asyncio
import enum
import io
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
def set_content_disposition(self, disptype: str, quote_fields: bool=True, _charset: str='utf-8', **params: Any) -> None:
    """Sets ``Content-Disposition`` header."""
    self._headers[hdrs.CONTENT_DISPOSITION] = content_disposition_header(disptype, quote_fields=quote_fields, _charset=_charset, **params)