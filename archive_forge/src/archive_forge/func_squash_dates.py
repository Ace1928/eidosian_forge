import math
import numbers
import re
import types
import warnings
from binascii import b2a_base64
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional, Union
from dateutil.parser import parse as _dateutil_parse
from dateutil.tz import tzlocal
def squash_dates(obj: Any) -> Any:
    """squash datetime objects into ISO8601 strings"""
    if isinstance(obj, dict):
        obj = dict(obj)
        for k, v in obj.items():
            obj[k] = squash_dates(v)
    elif isinstance(obj, (list, tuple)):
        obj = [squash_dates(o) for o in obj]
    elif isinstance(obj, datetime):
        obj = obj.isoformat()
    return obj