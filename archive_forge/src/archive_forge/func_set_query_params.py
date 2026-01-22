from __future__ import annotations
import urllib.parse as parse
from typing import Any
from streamlit import util
from streamlit.constants import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
@gather_metrics('experimental_set_query_params')
def set_query_params(**query_params: Any) -> None:
    """Set the query parameters that are shown in the browser's URL bar.

    .. warning::
        Query param `embed` cannot be set using this method.

    Parameters
    ----------
    **query_params : dict
        The query parameters to set, as key-value pairs.

    Example
    -------

    To point the user's web browser to something like
    "http://localhost:8501/?show_map=True&selected=asia&selected=america",
    you would do the following:

    >>> import streamlit as st
    >>>
    >>> st.experimental_set_query_params(
    ...     show_map=True,
    ...     selected=["asia", "america"],
    ... )

    """
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    ctx.mark_experimental_query_params_used()
    msg = ForwardMsg()
    msg.page_info_changed.query_string = _ensure_no_embed_params(query_params, ctx.query_string)
    ctx.query_string = msg.page_info_changed.query_string
    ctx.enqueue(msg)