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
def move_active(self, x, y):
    active = self.ActiveVertex
    if self.lock_var.get():
        x0, y0 = active.point()
        active.x, active.y = (float(x), float(y))
        if self.move_is_ok():
            if not self.generic_vertex(active):
                active.x, active.y = (x0, y0)
                if self.cursor_attached:
                    self.detach_cursor('non-generic active vertex')
                self.canvas.delete('lock_error')
                delta = 6
                self.canvas.create_oval(x0 - delta, y0 - delta, x0 + delta, y0 + delta, outline='gray', fill=None, width=3, tags='lock_error')
                return
            if not self.verify_drag():
                active.x, active.y = (x0, y0)
                if self.cursor_attached:
                    self.detach_cursor('non-generic diagram')
                return
            if not self.cursor_attached:
                self.attach_cursor('move is ok')
        else:
            if self.cursor_attached:
                self.detach_cursor('bad move')
            active.x, active.y = (x0, y0)
            self.ActiveVertex.draw()
            return
        self.canvas.delete('lock_error')
    else:
        active.x, active.y = (float(x), float(y))
    self.ActiveVertex.draw()
    if self.LiveArrow1:
        x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow1)
        self.canvas.coords(self.LiveArrow1, x0, y0, x, y)
    if self.LiveArrow2:
        x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow2)
        self.canvas.coords(self.LiveArrow2, x0, y0, x, y)
    self.update_smooth()
    self.update_info()
    self.window.update_idletasks()