from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def load_parent(self):
    """Provide TreeNode with a parent for the current node.  This function
        is only required if the tree was instantiated from a child node
        (virtual function)"""
    raise TreeWidgetError('virtual function.  Implement in subclass')