import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def selection_includes(self, *args, **kwargs):
    return self._listboxes[0].selection_includes(*args, **kwargs)