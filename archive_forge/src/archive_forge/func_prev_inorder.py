from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def prev_inorder(self) -> TreeWidget | None:
    """Return the previous TreeWidget depth first from this one."""
    this_node = self._node
    prev_node = this_node.prev_sibling()
    if prev_node is not None:
        prev_widget = prev_node.get_widget()
        last_child = prev_widget.last_child()
        if last_child is None:
            return prev_widget
        return last_child
    depth = this_node.get_depth()
    if prev_node is None and depth == 0:
        return None
    if prev_node is None:
        prev_node = this_node.get_parent()
    return prev_node.get_widget()