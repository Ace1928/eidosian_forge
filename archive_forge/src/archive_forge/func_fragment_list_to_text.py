from __future__ import annotations
from typing import Iterable, cast
from prompt_toolkit.utils import get_cwidth
from .base import (
def fragment_list_to_text(fragments: StyleAndTextTuples) -> str:
    """
    Concatenate all the text parts again.

    :param fragments: List of ``(style_str, text)`` or
        ``(style_str, text, mouse_handler)`` tuples.
    """
    ZeroWidthEscape = '[ZeroWidthEscape]'
    return ''.join((item[1] for item in fragments if ZeroWidthEscape not in item[0]))