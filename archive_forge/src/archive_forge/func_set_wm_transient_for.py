from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_transient_for(self, window, onerror=None):
    self.change_property(Xatom.WM_TRANSIENT_FOR, Xatom.WINDOW, 32, window.id, onerror=onerror)