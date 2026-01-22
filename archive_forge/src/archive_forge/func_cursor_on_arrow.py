import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def cursor_on_arrow(self, point):
    if self.lock_var.get():
        return False
    for arrow in self.Arrows:
        if arrow.too_close(point):
            return True
    return False