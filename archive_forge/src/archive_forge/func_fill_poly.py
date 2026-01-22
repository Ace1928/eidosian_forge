from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def fill_poly(self, gc, shape, coord_mode, points, onerror=None):
    request.FillPoly(display=self.display, onerror=onerror, shape=shape, coord_mode=coord_mode, drawable=self.id, gc=gc, points=points)