from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def delete_property(self, property, onerror=None):
    request.DeleteProperty(display=self.display, onerror=onerror, window=self.id, property=property)