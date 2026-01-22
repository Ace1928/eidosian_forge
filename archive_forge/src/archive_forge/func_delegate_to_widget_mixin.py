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
def delegate_to_widget_mixin(attribute_name: str) -> type[Widget]:
    """
    Return a mixin class that delegates all standard widget methods
    to an attribute given by attribute_name.

    This mixin is designed to be used as a superclass of another widget.
    """
    get_delegate = attrgetter(attribute_name)

    class DelegateToWidgetMixin(Widget):
        no_cache: typing.ClassVar[list[str]] = ['rows']

        def render(self, size, focus: bool=False) -> CompositeCanvas:
            canv = get_delegate(self).render(size, focus=focus)
            return CompositeCanvas(canv)

        @property
        def selectable(self) -> Callable[[], bool]:
            return get_delegate(self).selectable

        @property
        def get_cursor_coords(self) -> Callable[[tuple[()] | tuple[int] | tuple[int, int]], tuple[int, int] | None]:
            return get_delegate(self).get_cursor_coords

        @property
        def get_pref_col(self) -> Callable[[tuple[()] | tuple[int] | tuple[int, int]], int | None]:
            return get_delegate(self).get_pref_col

        def keypress(self, size: tuple[()] | tuple[int] | tuple[int, int], key: str) -> str | None:
            return get_delegate(self).keypress(size, key)

        @property
        def move_cursor_to_coords(self) -> Callable[[[tuple[()] | tuple[int] | tuple[int, int], int, int]], bool]:
            return get_delegate(self).move_cursor_to_coords

        @property
        def rows(self) -> Callable[[tuple[int], bool], int]:
            return get_delegate(self).rows

        @property
        def mouse_event(self) -> Callable[[tuple[()] | tuple[int] | tuple[int, int], str, int, int, int, bool], bool | None]:
            return get_delegate(self).mouse_event

        @property
        def sizing(self) -> Callable[[], frozenset[Sizing]]:
            return get_delegate(self).sizing

        @property
        def pack(self) -> Callable[[tuple[()] | tuple[int] | tuple[int, int], bool], tuple[int, int]]:
            return get_delegate(self).pack
    return DelegateToWidgetMixin