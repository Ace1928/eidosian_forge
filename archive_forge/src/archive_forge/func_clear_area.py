from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def clear_area(self, x=0, y=0, width=0, height=0, exposures=0, onerror=None):
    request.ClearArea(display=self.display, onerror=onerror, exposures=exposures, window=self.id, x=x, y=y, width=width, height=height)