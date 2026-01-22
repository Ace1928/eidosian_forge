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
def destroy_arrow(self, arrow):
    self.Arrows.remove(arrow)
    if arrow.end:
        arrow.end.in_arrow = None
    if arrow.start:
        arrow.start.out_arrow = None
    arrow.erase()
    self.Crossings = [c for c in self.Crossings if arrow not in c]