import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
def showtip(self, text):
    """Display text in tooltip window."""
    self.text = text
    if self.tipwindow or not self.text:
        return
    x, y, _, _ = self.widget.bbox('insert')
    x = x + self.widget.winfo_rootx() + self.widget.winfo_width()
    y = y + self.widget.winfo_rooty()
    self.tipwindow = tw = tk.Toplevel(self.widget)
    tw.wm_overrideredirect(1)
    tw.wm_geometry('+%d+%d' % (x, y))
    try:
        tw.tk.call('::tk::unsupported::MacWindowStyle', 'style', tw._w, 'help', 'noActivates')
    except tk.TclError:
        pass
    label = tk.Label(tw, text=self.text, justify=tk.LEFT, relief=tk.SOLID, borderwidth=1)
    label.pack(ipadx=1)