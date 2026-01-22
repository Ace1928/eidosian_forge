from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
@focus_item.setter
def focus_item(self, new_item):
    warnings.warn('only for backwards compatibility.You should use the new standard container properties `focus` and `focus_position` to get the child widget in focus or modify the focus position.', DeprecationWarning, stacklevel=2)
    self.focus = new_item