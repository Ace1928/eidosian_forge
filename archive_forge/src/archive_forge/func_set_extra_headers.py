from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
def set_extra_headers(self, path: str) -> None:
    """Disable cache for HTML files.

        Other assets like JS and CSS are suffixed with their hash, so they can
        be cached indefinitely.
        """
    is_index_url = len(path) == 0
    if is_index_url or path.endswith('.html'):
        self.set_header('Cache-Control', 'no-cache')
    else:
        self.set_header('Cache-Control', 'public')