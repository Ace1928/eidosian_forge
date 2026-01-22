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
def change_focus(self, size: tuple[int, int], position, offset_inset: int=0, coming_from: Literal['above', 'below'] | None=None, cursor_coords: tuple[int, int] | None=None, snap_rows: int | None=None) -> None:
    """
        Change the current focus widget.
        This is used internally by methods that know the widget's *size*.

        See also :meth:`.set_focus`.

        :param size: see :meth:`Widget.render` for details
        :param position: a position compatible with :meth:`self._body.set_focus`
        :param offset_inset: either the number of rows between the
            top of the listbox and the start of the focus widget (+ve
            value) or the number of lines of the focus widget hidden off
            the top edge of the listbox (-ve value) or 0 if the top edge
            of the focus widget is aligned with the top edge of the
            listbox (default if unspecified)
        :type offset_inset: int
        :param coming_from: either 'above', 'below' or unspecified `None`
        :type coming_from: str
        :param cursor_coords: (x, y) tuple indicating the desired
            column and row for the cursor, a (x,) tuple indicating only
            the column for the cursor, or unspecified
        :type cursor_coords: (int, int)
        :param snap_rows: the maximum number of extra rows to scroll
            when trying to "snap" a selectable focus into the view
        :type snap_rows: int
        """
    maxcol, maxrow = size
    if cursor_coords:
        self.pref_col = cursor_coords[0]
    else:
        self.update_pref_col_from_focus((maxcol, maxrow))
    self._invalidate()
    self._body.set_focus(position)
    target, _ignore = self._body.get_focus()
    tgt_rows = target.rows((maxcol,), True)
    if snap_rows is None:
        snap_rows = maxrow - 1
    align_top = 0
    align_bottom = maxrow - tgt_rows
    if coming_from == 'above' and target.selectable() and (offset_inset > align_bottom):
        if snap_rows >= offset_inset - align_bottom:
            offset_inset = align_bottom
        elif snap_rows >= offset_inset - align_top:
            offset_inset = align_top
        else:
            offset_inset -= snap_rows
    if coming_from == 'below' and target.selectable() and (offset_inset < align_top):
        if snap_rows >= align_top - offset_inset:
            offset_inset = align_top
        elif snap_rows >= align_bottom - offset_inset:
            offset_inset = align_bottom
        else:
            offset_inset += snap_rows
    if offset_inset >= 0:
        self.offset_rows = offset_inset
        self.inset_fraction = (0, 1)
    else:
        if offset_inset + tgt_rows <= 0:
            raise ListBoxError(f'Invalid offset_inset: {offset_inset}, only {tgt_rows} rows in target!')
        self.offset_rows = 0
        self.inset_fraction = (-offset_inset, tgt_rows)
    if cursor_coords is None:
        if coming_from is None:
            return
        cursor_coords = (self.pref_col,)
    if not hasattr(target, 'move_cursor_to_coords'):
        return
    attempt_rows = []
    if len(cursor_coords) == 1:
        pref_col, = cursor_coords
        if coming_from == 'above':
            attempt_rows = range(0, tgt_rows)
        else:
            if coming_from != 'below':
                raise ValueError("must specify coming_from ('above' or 'below') if cursor row is not specified")
            attempt_rows = range(tgt_rows, -1, -1)
    else:
        pref_col, pref_row = cursor_coords
        if pref_row < 0 or pref_row >= tgt_rows:
            raise ListBoxError(f'cursor_coords row outside valid range for target. pref_row:{pref_row!r} target_rows:{tgt_rows!r}')
        if coming_from == 'above':
            attempt_rows = range(pref_row, -1, -1)
        elif coming_from == 'below':
            attempt_rows = range(pref_row, tgt_rows)
        else:
            attempt_rows = [pref_row]
    for row in attempt_rows:
        if target.move_cursor_to_coords((maxcol,), pref_col, row):
            break