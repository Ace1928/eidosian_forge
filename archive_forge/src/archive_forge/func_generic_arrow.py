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
def generic_arrow(self, arrow):
    if arrow == None:
        return True
    locked = self.lock_var.get()
    for vertex in self.Vertices:
        if arrow.too_close(vertex):
            if locked:
                x, y, delta = (vertex.x, vertex.y, 6)
                self.canvas.delete('lock_error')
                self.canvas.create_oval(x - delta, y - delta, x + delta, y + delta, outline='gray', fill=None, width=3, tags='lock_error')
            return False
    for crossing in self.Crossings:
        point = self.CrossPoints[self.Crossings.index(crossing)]
        if arrow not in crossing and arrow.too_close(point):
            if locked:
                x, y, delta = (point.x, point.y, 6)
                self.canvas.delete('lock_error')
                self.canvas.create_oval(x - delta, y - delta, x + delta, y + delta, outline='gray', fill=None, width=3, tags='lock_error')
            return False
    return True