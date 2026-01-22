from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def is_mem_address_str(string):
    """Returns True if the string looks like <foo blarg at 0x15ee6f9a0>."""
    if _OBJ_MEM_ADDRESS.match(string):
        return True
    return False