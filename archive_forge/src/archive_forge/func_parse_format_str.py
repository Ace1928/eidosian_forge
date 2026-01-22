import ctypes
import re
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def parse_format_str(format_str, data):
    """Parse datetime `format_str` to interpret the `data`."""
    timestamp_meta = re.match('ts([smun]):(.*)', format_str)
    if timestamp_meta:
        unit, tz = (timestamp_meta.group(1), timestamp_meta.group(2))
        if tz != '':
            raise NotImplementedError('Timezones are not supported yet')
        if unit != 's':
            unit += 's'
        data = data.astype(f'datetime64[{unit}]')
        return data
    date_meta = re.match('td([Dm])', format_str)
    if date_meta:
        unit = date_meta.group(1)
        if unit == 'D':
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype('datetime64[s]')
        elif unit == 'm':
            data = data.astype('datetime64[ms]')
        else:
            raise NotImplementedError(f'Date unit is not supported: {unit}')
        return data
    raise NotImplementedError(f'DateTime kind is not supported: {format_str}')