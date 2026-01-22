from __future__ import annotations
import contextlib
import hashlib
import threading
import types
from dataclasses import dataclass
from typing import (
from google.protobuf.message import Message
import streamlit as st
from streamlit import runtime, util
from streamlit.elements import NONWIDGET_ELEMENTS, WIDGETS
from streamlit.logger import get_logger
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.hashing import update_hash
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.util import HASHLIB_KWARGS
@contextlib.contextmanager
def calling_cached_function(self, func: types.FunctionType, allow_widgets: bool) -> Iterator[None]:
    """Context manager that should wrap the invocation of a cached function.
        It allows us to track any `st.foo` messages that are generated from inside the function
        for playback during cache retrieval.
        """
    self._cached_func_stack.append(func)
    self._cached_message_stack.append([])
    self._seen_dg_stack.append(set())
    if allow_widgets:
        self._allow_widgets += 1
    try:
        yield
    finally:
        self._cached_func_stack.pop()
        self._most_recent_messages = self._cached_message_stack.pop()
        self._seen_dg_stack.pop()
        if allow_widgets:
            self._allow_widgets -= 1