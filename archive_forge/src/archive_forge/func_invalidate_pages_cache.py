from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Any, Callable, Final, cast
from blinker import Signal
from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5
def invalidate_pages_cache() -> None:
    global _cached_pages
    _LOGGER.debug('Pages directory changed')
    with _pages_cache_lock:
        _cached_pages = None
    _on_pages_changed.send()