import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def scan_dragto(self, *args, **kwargs):
    for lb in self._listboxes:
        lb.scan_dragto(*args, **kwargs)