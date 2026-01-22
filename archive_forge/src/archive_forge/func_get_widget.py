from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_widget(self, reload: bool=False) -> TreeWidget:
    """Return the widget for this node."""
    if self._widget is None or reload:
        self._widget = self.load_widget()
    return self._widget