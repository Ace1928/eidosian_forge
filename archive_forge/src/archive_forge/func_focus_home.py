from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def focus_home(self, size: tuple[int, int]) -> None:
    """Move focus to very top."""
    _widget, pos = self.body.get_focus()
    rootnode = pos.get_root()
    self.change_focus(size, rootnode)