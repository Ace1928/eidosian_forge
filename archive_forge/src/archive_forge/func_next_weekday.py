import datetime
from typing import Any, Optional, cast
from dateutil.rrule import WEEKLY, rrule
from arrow.constants import (
def next_weekday(start_date: Optional[datetime.date], weekday: int) -> datetime.datetime:
    """Get next weekday from the specified start date.

    :param start_date: Datetime object representing the start date.
    :param weekday: Next weekday to obtain. Can be a value between 0 (Monday) and 6 (Sunday).
    :return: Datetime object corresponding to the next weekday after start_date.

    Usage::

        # Get first Monday after epoch
        >>> next_weekday(datetime(1970, 1, 1), 0)
        1970-01-05 00:00:00

        # Get first Thursday after epoch
        >>> next_weekday(datetime(1970, 1, 1), 3)
        1970-01-01 00:00:00

        # Get first Sunday after epoch
        >>> next_weekday(datetime(1970, 1, 1), 6)
        1970-01-04 00:00:00
    """
    if weekday < 0 or weekday > 6:
        raise ValueError('Weekday must be between 0 (Monday) and 6 (Sunday).')
    return cast(datetime.datetime, rrule(freq=WEEKLY, dtstart=start_date, byweekday=weekday, count=1)[0])