from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def unmark(self, item=None):
    """
        Remove highlighting from the given item; or from every item,
        if no item is given.
        :raise ValueError: If ``item`` is not contained in the list.
        :raise KeyError: If ``item`` is not marked.
        """
    if item is None:
        self._marks.clear()
        self._textwidget.tag_remove('highlight', '1.0', 'end+1char')
    else:
        index = self._items.index(item)
        del self._marks[item]
        start, end = ('%d.0' % (index + 1), '%d.0' % (index + 2))
        self._textwidget.tag_remove('highlight', start, end)