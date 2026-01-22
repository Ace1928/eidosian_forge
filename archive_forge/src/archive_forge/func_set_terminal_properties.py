from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def set_terminal_properties(self, colors: int | None=None, bright_is_bold: bool | None=None, has_underline: bool | None=None) -> None:
    if colors is None:
        colors = self.colors
    if bright_is_bold is None:
        bright_is_bold = self.bright_is_bold
    if has_underline is None:
        has_underline = self.has_underline
    self.colors = colors
    self.bright_is_bold = bright_is_bold
    self.has_underline = has_underline