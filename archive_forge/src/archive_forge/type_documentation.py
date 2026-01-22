from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
Message representing a single field of a struct.

        Attributes:
            name (str):
                The name of the field. For reads, this is the column name.
                For SQL queries, it is the column alias (e.g., ``"Word"`` in
                the query ``"SELECT 'hello' AS Word"``), or the column name
                (e.g., ``"ColName"`` in the query
                ``"SELECT ColName FROM Table"``). Some columns might have an
                empty name (e.g., ``"SELECT UPPER(ColName)"``). Note that a
                query result can contain multiple fields with the same name.
            type_ (googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.Type):
                The type of the field.
        