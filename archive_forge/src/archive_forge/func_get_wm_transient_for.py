from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_transient_for(self):
    d = self.get_property(Xatom.WM_TRANSIENT_FOR, Xatom.WINDOW, 0, 1)
    if d is None or d.format != 32 or len(d.value) < 1:
        return None
    else:
        cls = self.display.get_resource_class('window', Window)
        return cls(self.display, d.value[0])