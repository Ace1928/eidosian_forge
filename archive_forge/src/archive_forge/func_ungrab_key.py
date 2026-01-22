from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def ungrab_key(self, key, modifiers, onerror=None):
    request.UngrabKey(display=self.display, onerror=onerror, key=key, grab_window=self.id, modifiers=modifiers)