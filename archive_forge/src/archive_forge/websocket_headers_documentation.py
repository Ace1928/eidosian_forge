from __future__ import annotations
from streamlit import runtime
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
Return a copy of the HTTP request headers for the current session's
    WebSocket connection. If there's no active session, return None instead.

    Raise an error if the server is not running.

    Note to the intrepid: this is an UNSUPPORTED, INTERNAL API. (We don't have plans
    to remove it without a replacement, but we don't consider this a production-ready
    function, and its signature may change without a deprecation warning.)
    