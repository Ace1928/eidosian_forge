from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def convert_selection(self, selection, target, property, time, onerror=None):
    request.ConvertSelection(display=self.display, onerror=onerror, requestor=self.id, selection=selection, target=target, property=property, time=time)