import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def selected_row(self):
    """
        Return the index of the currently selected row, or None if
        no row is selected.  To get the row value itself, use
        ``table[table.selected_row()]``.
        """
    sel = self._mlb.curselection()
    if sel:
        return int(sel[0])
    else:
        return None