from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def register_palette(self, palette: Iterable[tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, str] | tuple[str, str, str, str, str, str]]) -> None:
    """Register a set of palette entries.

        palette -- a list of (name, like_other_name) or
        (name, foreground, background, mono, foreground_high, background_high) tuples

            The (name, like_other_name) format will copy the settings
            from the palette entry like_other_name, which must appear
            before this tuple in the list.

            The mono and foreground/background_high values are
            optional ie. the second tuple format may have 3, 4 or 6
            values.  See register_palette_entry() for a description
            of the tuple values.
        """
    for item in palette:
        if len(item) in {3, 4, 6}:
            self.register_palette_entry(*item)
            continue
        if len(item) != 2:
            raise ScreenError(f'Invalid register_palette entry: {item!r}')
        name, like_name = item
        if like_name not in self._palette:
            raise ScreenError(f"palette entry '{like_name}' doesn't exist")
        self._palette[name] = self._palette[like_name]