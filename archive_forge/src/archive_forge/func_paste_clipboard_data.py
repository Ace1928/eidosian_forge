from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def paste_clipboard_data(self, data, paste_mode=PasteMode.EMACS, count=1):
    """
        Return a new :class:`.Document` instance which contains the result if
        we would paste this data at the current cursor position.

        :param paste_mode: Where to paste. (Before/after/emacs.)
        :param count: When >1, Paste multiple times.
        """
    assert isinstance(data, ClipboardData)
    assert paste_mode in (PasteMode.VI_BEFORE, PasteMode.VI_AFTER, PasteMode.EMACS)
    before = paste_mode == PasteMode.VI_BEFORE
    after = paste_mode == PasteMode.VI_AFTER
    if data.type == SelectionType.CHARACTERS:
        if after:
            new_text = self.text[:self.cursor_position + 1] + data.text * count + self.text[self.cursor_position + 1:]
        else:
            new_text = self.text_before_cursor + data.text * count + self.text_after_cursor
        new_cursor_position = self.cursor_position + len(data.text) * count
        if before:
            new_cursor_position -= 1
    elif data.type == SelectionType.LINES:
        l = self.cursor_position_row
        if before:
            lines = self.lines[:l] + [data.text] * count + self.lines[l:]
            new_text = '\n'.join(lines)
            new_cursor_position = len(''.join(self.lines[:l])) + l
        else:
            lines = self.lines[:l + 1] + [data.text] * count + self.lines[l + 1:]
            new_cursor_position = len(''.join(self.lines[:l + 1])) + l + 1
            new_text = '\n'.join(lines)
    elif data.type == SelectionType.BLOCK:
        lines = self.lines[:]
        start_line = self.cursor_position_row
        start_column = self.cursor_position_col + (0 if before else 1)
        for i, line in enumerate(data.text.split('\n')):
            index = i + start_line
            if index >= len(lines):
                lines.append('')
            lines[index] = lines[index].ljust(start_column)
            lines[index] = lines[index][:start_column] + line * count + lines[index][start_column:]
        new_text = '\n'.join(lines)
        new_cursor_position = self.cursor_position + (0 if before else 1)
    return Document(text=new_text, cursor_position=new_cursor_position)