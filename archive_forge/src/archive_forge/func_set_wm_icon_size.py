from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_icon_size(self, hints={}, onerror=None, **keys):
    self._set_struct_prop(Xatom.WM_ICON_SIZE, Xatom.WM_ICON_SIZE, icccm.WMIconSize, hints, keys, onerror)