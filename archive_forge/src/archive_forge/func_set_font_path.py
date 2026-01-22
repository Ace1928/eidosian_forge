import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def set_font_path(self, path, onerror=None):
    """Set the font path to path, which should be a list of strings.
        If path is empty, the default font path of the server will be
        restored."""
    request.SetFontPath(display=self.display, onerror=onerror, path=path)