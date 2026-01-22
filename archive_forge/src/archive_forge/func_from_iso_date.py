from __future__ import annotations
import collections
import datetime as dt
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Mapping
from email.utils import format_datetime, parsedate_to_datetime
from pprint import pprint as py_pprint
from marshmallow.base import FieldABC
from marshmallow.exceptions import FieldInstanceResolutionError
from marshmallow.warnings import RemovedInMarshmallow4Warning
def from_iso_date(value):
    """Parse a string and return a datetime.date."""
    match = _iso8601_date_re.match(value)
    if not match:
        raise ValueError('Not a valid ISO8601-formatted date string')
    kw = {k: int(v) for k, v in match.groupdict().items()}
    return dt.date(**kw)