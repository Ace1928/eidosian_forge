from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def copy_plane(self, gc, src_drawable, src_x, src_y, width, height, dst_x, dst_y, bit_plane, onerror=None):
    request.CopyPlane(display=self.display, onerror=onerror, src_drawable=src_drawable, dst_drawable=self.id, gc=gc, src_x=src_x, src_y=src_y, dst_x=dst_x, dst_y=dst_y, width=width, height=height, bit_plane=bit_plane)