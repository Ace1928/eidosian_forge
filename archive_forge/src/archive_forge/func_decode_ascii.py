from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def decode_ascii(string: bytes) -> str:
    """Decodes a string as ascii."""
    return string.decode('ascii')