import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def timestamp_pb(self):
    """Return a timestamp message.

        Returns:
            (:class:`~google.protobuf.timestamp_pb2.Timestamp`): Timestamp message
        """
    inst = self if self.tzinfo is not None else self.replace(tzinfo=datetime.timezone.utc)
    delta = inst - _UTC_EPOCH
    seconds = int(delta.total_seconds())
    nanos = self._nanosecond or self.microsecond * 1000
    return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)