from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.runtime.metrics_util import gather_metrics
def column_proto(normalized_weight: float) -> BlockProto:
    col_proto = BlockProto()
    col_proto.column.weight = normalized_weight
    col_proto.column.gap = gap_size
    col_proto.allow_empty = True
    return col_proto