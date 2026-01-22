from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_header_end(self) -> None:
    field = self._current_partial_header_name.lower()
    if field == b'content-disposition':
        self._current_part.content_disposition = self._current_partial_header_value
    self._current_part.item_headers.append((field, self._current_partial_header_value))
    self._current_partial_header_name = b''
    self._current_partial_header_value = b''