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
def zoom_to_fit(self):
    W, H = (self.canvas.winfo_width(), self.canvas.winfo_height())
    if W < 10:
        W, H = (self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight())
    x0, y0, x1, y1 = (W, H, 0, 0)
    for V in self.Vertices:
        x0, y0 = (min(x0, V.x), min(y0, V.y))
        x1, y1 = (max(x1, V.x), max(y1, V.y))
    w, h = (x1 - x0, y1 - y0)
    factor = min((W - 60) / w, (H - 60) / h)
    xfactor, yfactor = (round(factor * w) / w, round(factor * h) / h)
    self._zoom(xfactor, yfactor)
    try:
        x0, y0, x1, y1 = self.canvas.bbox('transformable')
        self._shift((W - x1 + x0) / 2 - x0, (H - y1 + y0) / 2 - y0)
    except TypeError:
        pass