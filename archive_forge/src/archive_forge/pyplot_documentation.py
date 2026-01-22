from __future__ import annotations
import io
from typing import TYPE_CHECKING, Any, cast
import streamlit.elements.image as image_utils
from streamlit import config
from streamlit.errors import StreamlitDeprecationWarning
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime.metrics_util import gather_metrics
Get our DeltaGenerator.