import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def pending_events(self):
    """Return the number of events queued, i.e. the number of times
        that Display.next_event() can be called without blocking."""
    return self.display.pending_events()