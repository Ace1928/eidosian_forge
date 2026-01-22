from __future__ import annotations
import typing
import warnings
from urwid.split_repr import remove_defaults
from .columns import Columns
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .divider import Divider
from .monitored_list import MonitoredFocusList, MonitoredList
from .padding import Padding
from .pile import Pile
from .widget import Widget, WidgetError, WidgetWarning, WidgetWrap
def get_display_widget(self, size: tuple[int] | tuple[()]) -> Divider | Pile:
    """
        Arrange the cells into columns (and possibly a pile) for
        display, input or to calculate rows, and update the display
        widget.
        """
    maxcol = self._get_maxcol(size)
    if self._cache_maxcol == maxcol:
        return self._w
    self._cache_maxcol = maxcol
    self._w = self.generate_display_widget((maxcol,))
    return self._w