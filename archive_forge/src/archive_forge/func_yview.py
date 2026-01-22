import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def yview(self, *args, **kwargs):
    for lb in self._listboxes:
        v = lb.yview(*args, **kwargs)
    return v