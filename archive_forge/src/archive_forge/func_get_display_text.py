from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_display_text(self) -> str | tuple[Hashable, str] | list[str | tuple[Hashable, str]]:
    return f'{self.get_node().get_key()}: {self.get_node().get_value()!s}'