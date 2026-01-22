from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import keys
Arguments to [delete][google.spanner.v1.Mutation.delete] operations.

        Attributes:
            table (str):
                Required. The table whose rows will be
                deleted.
            key_set (googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.KeySet):
                Required. The primary keys of the rows within
                [table][google.spanner.v1.Mutation.Delete.table] to delete.
                The primary keys must be specified in the order in which
                they appear in the ``PRIMARY KEY()`` clause of the table's
                equivalent DDL statement (the DDL statement used to create
                the table). Delete is idempotent. The transaction will
                succeed even if some or all rows do not exist.
        