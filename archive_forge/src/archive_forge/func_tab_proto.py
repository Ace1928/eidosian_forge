from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.runtime.metrics_util import gather_metrics
def tab_proto(label: str) -> BlockProto:
    tab_proto = BlockProto()
    tab_proto.tab.label = label
    tab_proto.allow_empty = True
    return tab_proto