from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def screenshot_collect() -> list[str]:
    """Return screenshots as a list of HTML fragments."""
    fragments, HtmlGenerator.fragments = (HtmlGenerator.fragments, [])
    return fragments