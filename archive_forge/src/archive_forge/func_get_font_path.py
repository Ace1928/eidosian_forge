import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_font_path(self):
    """Return the current font path as a list of strings."""
    r = request.GetFontPath(display=self.display)
    return r.paths