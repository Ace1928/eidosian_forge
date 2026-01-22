from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def translate_coords(self, src_window, src_x, src_y):
    return request.TranslateCoords(display=self.display, src_wid=src_window, dst_wid=self.id, src_x=src_x, src_y=src_y)