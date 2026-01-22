from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def is_hex_color_like(color: MaybeColor) -> bool:
    """Check whether the input looks like a hex color.

    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try
    to convert and see if an error is thrown.
    """
    return isinstance(color, str) and color.startswith('#') and color[1:].isalnum() and (len(color) in {4, 5, 7, 9})