from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import (
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import (
from xarray.core.utils import emit_user_level_warning
def to_cftime_datetime(date_str_or_date, calendar=None):
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if isinstance(date_str_or_date, str):
        if calendar is None:
            raise ValueError('If converting a string to a cftime.datetime object, a calendar type must be provided')
        date, _ = _parse_iso8601_with_reso(get_date_type(calendar), date_str_or_date)
        return date
    elif isinstance(date_str_or_date, cftime.datetime):
        return date_str_or_date
    elif isinstance(date_str_or_date, (datetime, pd.Timestamp)):
        return cftime.DatetimeProlepticGregorian(*date_str_or_date.timetuple())
    else:
        raise TypeError(f'date_str_or_date must be a string or a subclass of cftime.datetime. Instead got {date_str_or_date!r}.')