from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def to_int_color_tuple(color: MaybeColor) -> IntColorTuple:
    """Convert input into color tuple of type (int, int, int, int)."""
    color_tuple = _to_color_tuple(color, rgb_formatter=_int_formatter, alpha_formatter=_int_formatter)
    return cast(IntColorTuple, color_tuple)