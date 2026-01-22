from __future__ import annotations
import dataclasses
import inspect
import json
import types
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Final, Generator, Iterable, List, cast
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state import QueryParamsProxy, SessionStateProxy
from streamlit.string_util import (
from streamlit.user_info import UserInfoProxy
def flush_buffer():
    if string_buffer:
        text_content = ' '.join(string_buffer)
        text_container = self.dg.empty()
        text_container.markdown(text_content, unsafe_allow_html=unsafe_allow_html)
        string_buffer[:] = []