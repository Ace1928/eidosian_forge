from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def shift_focus(self, size: tuple[int, int], offset_inset: int) -> None:
    """
        Move the location of the current focus relative to the top.
        This is used internally by methods that know the widget's *size*.

        See also :meth:`.set_focus_valign`.

        :param size: see :meth:`Widget.render` for details
        :param offset_inset: either the number of rows between the
            top of the listbox and the start of the focus widget (+ve
            value) or the number of lines of the focus widget hidden off
            the top edge of the listbox (-ve value) or ``0`` if the top edge
            of the focus widget is aligned with the top edge of the
            listbox.
        :type offset_inset: int
        """
    maxcol, maxrow = size
    if offset_inset >= 0:
        if offset_inset >= maxrow:
            raise ListBoxError(f'Invalid offset_inset: {offset_inset!r}, only {maxrow!r} rows in list box')
        self.offset_rows = offset_inset
        self.inset_fraction = (0, 1)
    else:
        target, _ignore = self._body.get_focus()
        tgt_rows = target.rows((maxcol,), True)
        if offset_inset + tgt_rows <= 0:
            raise ListBoxError(f'Invalid offset_inset: {offset_inset!r}, only {tgt_rows!r} rows in target!')
        self.offset_rows = 0
        self.inset_fraction = (-offset_inset, tgt_rows)
    self._invalidate()