from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def image_text(self, gc, x, y, string, onerror=None):
    request.ImageText8(display=self.display, onerror=onerror, drawable=self.id, gc=gc, x=x, y=y, string=string)