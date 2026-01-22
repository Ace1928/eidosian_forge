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
@cell_width.setter
def cell_width(self, width: int) -> None:
    focus_position = self.focus_position
    self.contents = [(w, (WHSettings.GIVEN, width)) for w, options in self.contents]
    self.focus_position = focus_position
    self._cell_width = width