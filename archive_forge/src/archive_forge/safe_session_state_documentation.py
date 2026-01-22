from __future__ import annotations
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import RegisterWidgetResult, T, WidgetMetadata
from streamlit.runtime.state.query_params import QueryParams
from streamlit.runtime.state.session_state import SessionState
Presents itself as a simple dict of the underlying SessionState instance