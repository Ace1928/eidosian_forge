from __future__ import annotations
import contextlib
import os
def colored_map(text: str, cmap: dict) -> str:
    """
    Return colorized text. cmap is a dict mapping tokens to color options.

    colored_key("foo bar", {bar: "green"})
    colored_key("foo bar", {bar: {"color": "green", "on_color": "on_red"}})
    """
    if not __ISON:
        return text
    for key, v in cmap.items():
        if isinstance(v, dict):
            text = text.replace(key, colored(key, **v))
        else:
            text = text.replace(key, colored(key, color=v))
    return text