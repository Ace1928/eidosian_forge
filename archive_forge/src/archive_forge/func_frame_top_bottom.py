from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasCombine, CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, VAlign
from .container import WidgetContainerMixin
from .filler import Filler
from .widget import Widget, WidgetError
def frame_top_bottom(self, size: tuple[int, int], focus: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    """
        Calculate the number of rows for the header and footer.

        :param size: See :meth:`Widget.render` for details
        :type size: widget size
        :param focus: ``True`` if this widget is in focus
        :type focus: bool
        :returns: `(head rows, foot rows),(orig head, orig foot)`
                  orig head/foot are from rows() calls.
        :rtype: (int, int), (int, int)
        """
    maxcol, maxrow = size
    frows = hrows = 0
    if self.header:
        hrows = self.header.rows((maxcol,), self.focus_part == 'header' and focus)
    if self.footer:
        frows = self.footer.rows((maxcol,), self.focus_part == 'footer' and focus)
    remaining = maxrow
    if self.focus_part == 'footer':
        if frows >= remaining:
            return ((0, remaining), (hrows, frows))
        remaining -= frows
        if hrows >= remaining:
            return ((remaining, frows), (hrows, frows))
    elif self.focus_part == 'header':
        if hrows >= maxrow:
            return ((remaining, 0), (hrows, frows))
        remaining -= hrows
        if frows >= remaining:
            return ((hrows, remaining), (hrows, frows))
    elif hrows + frows >= remaining:
        rless1 = max(0, remaining - 1)
        if frows >= remaining - 1:
            return ((0, rless1), (hrows, frows))
        remaining -= frows
        rless1 = max(0, remaining - 1)
        return ((rless1, frows), (hrows, frows))
    return ((hrows, frows), (hrows, frows))