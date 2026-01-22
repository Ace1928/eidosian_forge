from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_hints(self):
    return self._get_struct_prop(Xatom.WM_HINTS, Xatom.WM_HINTS, icccm.WMHints)