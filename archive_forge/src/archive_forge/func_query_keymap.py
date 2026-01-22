import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def query_keymap(self):
    """Return a bit vector for the logical state of the keyboard,
        where each bit set to 1 indicates that the corresponding key is
        currently pressed down. The vector is represented as a list of 32
        integers. List item N contains the bits for keys 8N to 8N + 7
        with the least significant bit in the byte representing key 8N."""
    r = request.QueryKeymap(display=self.display)
    return r.map