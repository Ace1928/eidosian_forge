from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def validate_emoji(maybe_emoji: str | None) -> str:
    if maybe_emoji is None:
        return ''
    elif is_emoji(maybe_emoji):
        return maybe_emoji
    else:
        raise StreamlitAPIException(f'The value "{maybe_emoji}" is not a valid emoji. Shortcodes are not allowed, please use a single character instead.')