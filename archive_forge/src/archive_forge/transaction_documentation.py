from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
``ReadLockMode`` is used to set the read lock mode for read-write
            transactions.

            Values:
                READ_LOCK_MODE_UNSPECIFIED (0):
                    Default value.

                    If the value is not specified, the pessimistic
                    read lock is used.
                PESSIMISTIC (1):
                    Pessimistic lock mode.

                    Read locks are acquired immediately on read.
                OPTIMISTIC (2):
                    Optimistic lock mode.

                    Locks for reads within the transaction are not
                    acquired on read. Instead the locks are acquired
                    on a commit to validate that read/queried data
                    has not changed since the transaction started.
            