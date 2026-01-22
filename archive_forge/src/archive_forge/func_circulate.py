from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def circulate(self, direction, onerror=None):
    request.CirculateWindow(display=self.display, onerror=onerror, direction=direction, window=self.id)