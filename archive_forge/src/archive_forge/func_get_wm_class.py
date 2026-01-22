from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_class(self):
    d = self.get_full_property(Xatom.WM_CLASS, Xatom.STRING)
    if d is None or d.format != 8:
        return None
    else:
        parts = d.value.split('\x00')
        if len(parts) < 2:
            return None
        else:
            return (parts[0], parts[1])