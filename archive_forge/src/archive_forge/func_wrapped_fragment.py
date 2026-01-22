from __future__ import annotations
import contextlib
import hashlib
import inspect
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Protocol, TypeVar, overload
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.time_util import time_to_seconds
def wrapped_fragment():
    import streamlit as st
    ctx = get_script_run_ctx(suppress_warning=True)
    assert ctx is not None
    if ctx.fragment_ids_this_run:
        ctx.cursors = deepcopy(cursors_snapshot)
        dg_stack.set(deepcopy(dg_stack_snapshot))
    else:
        ctx.current_fragment_id = fragment_id
    try:
        with st.container():
            result = non_optional_func(*args, **kwargs)
    finally:
        ctx.current_fragment_id = None
    return result