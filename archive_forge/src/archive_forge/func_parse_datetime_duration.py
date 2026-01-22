from collections import OrderedDict
from decimal import Decimal, InvalidOperation
import arrow  # type: ignore
from isoduration.parser.exceptions import (
from isoduration.parser.util import (
from isoduration.types import DateDuration, Duration, TimeDuration
def parse_datetime_duration(duration_str: str, sign: int) -> Duration:
    try:
        duration: arrow.Arrow = arrow.get(duration_str)
    except (arrow.ParserError, ValueError):
        raise UnparseableValue(f'Value could not be parsed as datetime: {duration_str}')
    return Duration(DateDuration(years=sign * duration.year, months=sign * duration.month, days=sign * duration.day), TimeDuration(hours=sign * duration.hour, minutes=sign * duration.minute, seconds=sign * duration.second))