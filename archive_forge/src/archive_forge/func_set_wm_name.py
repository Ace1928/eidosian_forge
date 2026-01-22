from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_name(self, name, onerror=None):
    self.change_property(Xatom.WM_NAME, Xatom.STRING, 8, name, onerror=onerror)