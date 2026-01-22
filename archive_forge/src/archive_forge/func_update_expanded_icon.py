from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def update_expanded_icon(self) -> None:
    """Update display widget text for parent widgets"""
    icon = [self.unexpanded_icon, self.expanded_icon][self.expanded]
    self._w.base_widget.contents[0] = (icon, (WHSettings.GIVEN, 1, False))