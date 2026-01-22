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
def cache_widget_render(cls):
    """
    Return a function that wraps the cls.render() method
    and fetches and stores canvases with CanvasCache.
    """
    ignore_focus = bool(getattr(cls, 'ignore_focus', False))
    fn = cls.render

    @functools.wraps(fn)
    def cached_render(self, size, focus=False):
        focus = focus and (not ignore_focus)
        canv = CanvasCache.fetch(self, cls, size, focus)
        if canv:
            return canv
        canv = fn(self, size, focus=focus)
        validate_size(self, size, canv)
        if canv.widget_info:
            canv = CompositeCanvas(canv)
        canv.finalize(self, size, focus)
        CanvasCache.store(cls, canv)
        return canv
    cached_render.original_fn = fn
    return cached_render