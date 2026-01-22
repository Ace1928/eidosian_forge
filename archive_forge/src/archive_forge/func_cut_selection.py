from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def cut_selection(self):
    """
        Return a (:class:`.Document`, :class:`.ClipboardData`) tuple, where the
        document represents the new document when the selection is cut, and the
        clipboard data, represents whatever has to be put on the clipboard.
        """
    if self.selection:
        cut_parts = []
        remaining_parts = []
        new_cursor_position = self.cursor_position
        last_to = 0
        for from_, to in self.selection_ranges():
            if last_to == 0:
                new_cursor_position = from_
            remaining_parts.append(self.text[last_to:from_])
            cut_parts.append(self.text[from_:to + 1])
            last_to = to + 1
        remaining_parts.append(self.text[last_to:])
        cut_text = '\n'.join(cut_parts)
        remaining_text = ''.join(remaining_parts)
        if self.selection.type == SelectionType.LINES and cut_text.endswith('\n'):
            cut_text = cut_text[:-1]
        return (Document(text=remaining_text, cursor_position=new_cursor_position), ClipboardData(cut_text, self.selection.type))
    else:
        return (self, ClipboardData(''))