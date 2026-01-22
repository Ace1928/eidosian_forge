import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def to_milliseconds(value):
    """Convert a zone-aware datetime to milliseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Milliseconds since the unix epoch.
    """
    micros = to_microseconds(value)
    return micros // 1000