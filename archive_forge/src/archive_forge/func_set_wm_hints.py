from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_hints(self, hints={}, onerror=None, **keys):
    self._set_struct_prop(Xatom.WM_HINTS, Xatom.WM_HINTS, icccm.WMHints, hints, keys, onerror)