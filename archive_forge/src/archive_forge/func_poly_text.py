from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def poly_text(self, gc, x, y, items, onerror=None):
    request.PolyText8(display=self.display, onerror=onerror, drawable=self.id, gc=gc, x=x, y=y, items=items)