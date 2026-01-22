import re
import sys
from datetime import datetime, timedelta
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
from dateutil import tz
from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp
def parse_iso(self, datetime_string: str, normalize_whitespace: bool=False) -> datetime:
    if normalize_whitespace:
        datetime_string = re.sub('\\s+', ' ', datetime_string.strip())
    has_space_divider = ' ' in datetime_string
    has_t_divider = 'T' in datetime_string
    num_spaces = datetime_string.count(' ')
    if has_space_divider and num_spaces != 1 or (has_t_divider and num_spaces > 0):
        raise ParserError(f'Expected an ISO 8601-like string, but was given {datetime_string!r}. Try passing in a format string to resolve this.')
    has_time = has_space_divider or has_t_divider
    has_tz = False
    formats = ['YYYY-MM-DD', 'YYYY-M-DD', 'YYYY-M-D', 'YYYY/MM/DD', 'YYYY/M/DD', 'YYYY/M/D', 'YYYY.MM.DD', 'YYYY.M.DD', 'YYYY.M.D', 'YYYYMMDD', 'YYYY-DDDD', 'YYYYDDDD', 'YYYY-MM', 'YYYY/MM', 'YYYY.MM', 'YYYY', 'W']
    if has_time:
        if has_space_divider:
            date_string, time_string = datetime_string.split(' ', 1)
        else:
            date_string, time_string = datetime_string.split('T', 1)
        time_parts = re.split('[\\+\\-Z]', time_string, maxsplit=1, flags=re.IGNORECASE)
        time_components: Optional[Match[str]] = self._TIME_RE.match(time_parts[0])
        if time_components is None:
            raise ParserError('Invalid time component provided. Please specify a format or provide a valid time component in the basic or extended ISO 8601 time format.')
        hours, minutes, seconds, subseconds_sep, subseconds = time_components.groups()
        has_tz = len(time_parts) == 2
        has_minutes = minutes is not None
        has_seconds = seconds is not None
        has_subseconds = subseconds is not None
        is_basic_time_format = ':' not in time_parts[0]
        tz_format = 'Z'
        if has_tz and ':' in time_parts[1]:
            tz_format = 'ZZ'
        time_sep = '' if is_basic_time_format else ':'
        if has_subseconds:
            time_string = 'HH{time_sep}mm{time_sep}ss{subseconds_sep}S'.format(time_sep=time_sep, subseconds_sep=subseconds_sep)
        elif has_seconds:
            time_string = 'HH{time_sep}mm{time_sep}ss'.format(time_sep=time_sep)
        elif has_minutes:
            time_string = f'HH{time_sep}mm'
        else:
            time_string = 'HH'
        if has_space_divider:
            formats = [f'{f} {time_string}' for f in formats]
        else:
            formats = [f'{f}T{time_string}' for f in formats]
    if has_time and has_tz:
        formats = [f'{f}{tz_format}' for f in formats]
    return self._parse_multiformat(datetime_string, formats)