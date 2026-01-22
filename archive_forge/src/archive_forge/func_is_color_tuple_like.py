from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def is_color_tuple_like(color: MaybeColor) -> bool:
    """Check whether the input looks like a tuple color.

    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try
    to convert and see if an error is thrown.
    """
    return isinstance(color, (tuple, list)) and len(color) in {3, 4} and all((isinstance(c, (int, float)) for c in color))