from __future__ import annotations
import contextlib
import os
def cprint_map(text: str, cmap: dict, **kwargs) -> None:
    """
    Print colorize text.
    cmap is a dict mapping keys to color options.
    kwargs are passed to print function

    Examples:
        cprint_map("Hello world", {"Hello": "red"})
    """
    try:
        print(colored_map(text, cmap), **kwargs)
    except TypeError:
        kwargs.pop('flush', None)
        print(colored_map(text, cmap), **kwargs)