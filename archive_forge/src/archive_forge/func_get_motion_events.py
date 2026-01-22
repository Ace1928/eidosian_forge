from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_motion_events(self, start, stop):
    r = request.GetMotionEvents(display=self.display, window=self.id, start=start, stop=stop)
    return r.events