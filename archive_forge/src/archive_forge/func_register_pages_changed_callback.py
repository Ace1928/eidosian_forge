from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Any, Callable, Final, cast
from blinker import Signal
from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5
def register_pages_changed_callback(callback: Callable[[str], None]) -> Callable[[], None]:

    def disconnect():
        _on_pages_changed.disconnect(callback)
    _on_pages_changed.connect(callback, weak=False)
    return disconnect