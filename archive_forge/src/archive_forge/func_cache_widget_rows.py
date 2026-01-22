from __future__ import annotations
import functools
import logging
import typing
import warnings
from operator import attrgetter
from urwid import signals
from urwid.canvas import Canvas, CanvasCache, CompositeCanvas
from urwid.command_map import command_map
from urwid.split_repr import split_repr
from urwid.util import MetaSuper
from .constants import Sizing
def cache_widget_rows(cls):
    """
    Return a function that wraps the cls.rows() method
    and returns rows from the CanvasCache if available.
    """
    ignore_focus = bool(getattr(cls, 'ignore_focus', False))
    fn = cls.rows

    @functools.wraps(fn)
    def cached_rows(self, size: tuple[int], focus: bool=False) -> int:
        focus = focus and (not ignore_focus)
        canv = CanvasCache.fetch(self, cls, size, focus)
        if canv:
            return canv.rows()
        return fn(self, size, focus)
    return cached_rows