import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def rowconfigure(self, row_index, cnf={}, **kw):
    """:see: ``MultiListbox.rowconfigure()``"""
    self._mlb.rowconfigure(row_index, cnf, **kw)