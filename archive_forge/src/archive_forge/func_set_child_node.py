from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def set_child_node(self, key: Hashable, node: TreeNode) -> None:
    """Set the child node for a given key.

        Useful for bottom-up, lazy population of a tree.
        """
    self._children[key] = node