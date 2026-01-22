from __future__ import annotations
import datetime
from lazyops.imports._dateparser import (
from typing import Optional, List, Union
def parse_datetime_from_timestamp(timestamp: Optional[int]) -> Optional[datetime.datetime]:
    """
    Parses the timestamp into a datetime

    Format: 1666699200000 -> 2022-12-24T00:00:00.000Z
    """
    if timestamp is None:
        return None
    return datetime.datetime.fromtimestamp(timestamp / 1000, tz=datetime.timezone.utc)