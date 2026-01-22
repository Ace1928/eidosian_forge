from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def poly_arc(self, gc, arcs, onerror=None):
    request.PolyArc(display=self.display, onerror=onerror, drawable=self.id, gc=gc, arcs=arcs)