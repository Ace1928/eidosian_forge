from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def rotate_properties(self, properties, delta, onerror=None):
    request.RotateProperties(display=self.display, onerror=onerror, window=self.id, delta=delta, properties=properties)