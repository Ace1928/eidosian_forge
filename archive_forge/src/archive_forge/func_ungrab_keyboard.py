import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def ungrab_keyboard(self, time, onerror=None):
    """Ungrab a grabbed keyboard and any queued events. See
        XUngrabKeyboard(3X11)."""
    request.UngrabKeyboard(display=self.display, onerror=onerror, time=time)