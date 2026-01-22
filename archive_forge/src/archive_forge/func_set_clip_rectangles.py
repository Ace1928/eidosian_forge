from Xlib.protocol import request
from Xlib.xobject import resource, cursor
def set_clip_rectangles(self, x_origin, y_origin, rectangles, ordering, onerror=None):
    request.SetClipRectangles(display=self.display, onerror=onerror, ordering=ordering, gc=self.id, x_origin=x_origin, y_origin=y_origin, rectangles=rectangles)