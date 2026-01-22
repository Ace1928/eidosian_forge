import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def handle_down(self, event):
    if self.text.compare(Tk_.INSERT, '<', 'output_end'):
        return
    if self.editing_hist:
        insert_line = int(str(self.text.index(Tk_.INSERT)).split('.')[0])
        bottom_line = int(str(self.text.index('history_end')).split('.')[0])
        if insert_line < bottom_line:
            return
    if self.hist_pointer == 0:
        return
    self.text.delete('output_end', Tk_.END)
    self.hist_pointer -= 1
    if self.hist_pointer == 0:
        self.write(self.hist_stem.strip('\n'), style=(), advance=False)
        self.editing_hist = False
        self.text.tag_delete('history')
    else:
        self.write_history()
    return 'break'