import datetime
import errno
import os
import os.path
import time
def format_time_absolute(timestamp_pb):
    """Converts a `timestamp_pb2.Timestamp` to UTC time string.

    This will always be of the form "2001-02-03T04:05:06Z".

    Args:
      timestamp_pb: A `google.protobuf.timestamp_pb2.Timestamp` value to
        convert to string. The input will not be modified.

    Returns:
      An RFC 3339 date-time string.
    """
    dt = datetime.datetime.fromtimestamp(timestamp_pb.seconds, tz=datetime.timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')