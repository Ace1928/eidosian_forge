from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import TypeAlias
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.GraphVizChart_pb2 import GraphVizChart as GraphVizChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
Get our DeltaGenerator.