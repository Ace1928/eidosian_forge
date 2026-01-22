import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def can_render(string: str) -> bool:
    """Check if terminal can render unicode characters, e.g. special loading
    icons. Can be used to display fallbacks for ASCII terminals.

    string (str): The string to render.
    RETURNS (bool): Whether the terminal can render the text.
    """
    try:
        string.encode(ENCODING)
        return True
    except UnicodeEncodeError:
        return False