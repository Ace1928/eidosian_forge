from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
Response containing the cursor corresponding to a publish or
    event time in a topic partition.

    Attributes:
        cursor (google.cloud.pubsublite_v1.types.Cursor):
            If present, the cursor references the first message with
            time greater than or equal to the specified target time. If
            such a message cannot be found, the cursor will be unset
            (i.e. ``cursor`` is not present).
    