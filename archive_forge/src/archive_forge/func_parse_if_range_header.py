from __future__ import annotations
import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header
from ._internal import _dt_as_utc
from ._internal import _plain_int
from . import datastructures as ds
from .sansio import http as _sansio_http
def parse_if_range_header(value: str | None) -> ds.IfRange:
    """Parses an if-range header which can be an etag or a date.  Returns
    a :class:`~werkzeug.datastructures.IfRange` object.

    .. versionchanged:: 2.0
        If the value represents a datetime, it is timezone-aware.

    .. versionadded:: 0.7
    """
    if not value:
        return ds.IfRange()
    date = parse_date(value)
    if date is not None:
        return ds.IfRange(date=date)
    return ds.IfRange(unquote_etag(value)[0])