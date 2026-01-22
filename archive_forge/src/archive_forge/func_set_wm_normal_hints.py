from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_normal_hints(self, hints={}, onerror=None, **keys):
    self._set_struct_prop(Xatom.WM_NORMAL_HINTS, Xatom.WM_SIZE_HINTS, icccm.WMNormalHints, hints, keys, onerror)