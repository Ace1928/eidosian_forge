import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def yview_scroll(self, *args, **kwargs):
    for lb in self._listboxes:
        lb.yview_scroll(*args, **kwargs)