from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def prev_sibling(self) -> TreeNode | None:
    if self.get_depth() > 0:
        return self.get_parent().prev_child(self.get_key())
    return None