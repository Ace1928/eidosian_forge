from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def destroy_sub_windows(self, onerror=None):
    request.DestroySubWindows(display=self.display, onerror=onerror, window=self.id)