from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def move_focus_to_parent(self, size: tuple[int, int]) -> None:
    """Move focus to parent of widget in focus."""
    _widget, pos = self.body.get_focus()
    parentpos = pos.get_parent()
    if parentpos is None:
        return
    middle, top, _bottom = self.calculate_visible(size)
    row_offset, _focus_widget, _focus_pos, _focus_rows, _cursor = middle
    _trim_top, fill_above = top
    for _widget, pos, rows in fill_above:
        row_offset -= rows
        if pos == parentpos:
            self.change_focus(size, pos, row_offset)
            return
    self.change_focus(size, pos.get_parent())