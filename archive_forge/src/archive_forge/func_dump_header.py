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
def dump_header(iterable: dict[str, t.Any] | t.Iterable[t.Any]) -> str:
    """Produce a header value from a list of items or ``key=value`` pairs, separated by
    commas ``,``.

    This is the reverse of :func:`parse_list_header`, :func:`parse_dict_header`, and
    :func:`parse_set_header`.

    If a value contains non-token characters, it will be quoted.

    If a value is ``None``, the key is output alone.

    In some keys for some headers, a UTF-8 value can be encoded using a special
    ``key*=UTF-8''value`` form, where ``value`` is percent encoded. This function will
    not produce that format automatically, but if a given key ends with an asterisk
    ``*``, the value is assumed to have that form and will not be quoted further.

    .. code-block:: python

        dump_header(["foo", "bar baz"])
        'foo, "bar baz"'

        dump_header({"foo": "bar baz"})
        'foo="bar baz"'

    :param iterable: The items to create a header from.

    .. versionchanged:: 3.0
        The ``allow_token`` parameter is removed.

    .. versionchanged:: 2.2.3
        If a key ends with ``*``, its value will not be quoted.
    """
    if isinstance(iterable, dict):
        items = []
        for key, value in iterable.items():
            if value is None:
                items.append(key)
            elif key[-1] == '*':
                items.append(f'{key}={value}')
            else:
                items.append(f'{key}={quote_header_value(value)}')
    else:
        items = [quote_header_value(x) for x in iterable]
    return ', '.join(items)