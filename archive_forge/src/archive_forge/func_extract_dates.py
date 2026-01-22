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
def extract_dates(obj: Any) -> Any:
    """extract ISO8601 dates from unpacked JSON"""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = extract_dates(v)
        obj = new_obj
    elif isinstance(obj, (list, tuple)):
        obj = [extract_dates(o) for o in obj]
    elif isinstance(obj, str):
        obj = parse_date(obj)
    return obj