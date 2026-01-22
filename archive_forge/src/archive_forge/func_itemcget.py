import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def itemcget(self, *args, **kwargs):
    return self._listboxes[0].itemcget(*args, **kwargs)