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
def verify_drag(self):
    active = self.ActiveVertex
    active.update_arrows()
    self.update_crossings(active.in_arrow)
    self.update_crossings(active.out_arrow)
    self.update_crosspoints()
    return self.generic_arrow(active.in_arrow) and self.generic_arrow(active.out_arrow)