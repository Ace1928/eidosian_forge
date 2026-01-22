from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Any, Callable, Final, cast
from blinker import Signal
from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5
def open_python_file(filename: str):
    """Open a read-only Python file taking proper care of its encoding.

    In Python 3, we would like all files to be opened with utf-8 encoding.
    However, some author like to specify PEP263 headers in their source files
    with their own encodings. In that case, we should respect the author's
    encoding.
    """
    import tokenize
    if hasattr(tokenize, 'open'):
        return tokenize.open(filename)
    else:
        return open(filename, encoding='utf-8')