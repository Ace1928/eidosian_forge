from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
import urwid
from urwid.canvas import Canvas, CanvasJoin, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def set_focus_column(self, num: int) -> None:
    """
        Set the column in focus by its index in :attr:`widget_list`.

        :param num: index of focus-to-be entry
        :type num: int

        .. note:: only for backwards compatibility. You may also use the new
            standard container property :attr:`focus_position` to set the focus.
        """
    warnings.warn('only for backwards compatibility.You may also use the new standard container property `focus_position`', PendingDeprecationWarning, stacklevel=2)
    self.focus_position = num