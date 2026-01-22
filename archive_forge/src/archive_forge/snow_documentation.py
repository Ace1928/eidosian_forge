from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Snow_pb2 import Snow as SnowProto
from streamlit.runtime.metrics_util import gather_metrics
Get our DeltaGenerator.